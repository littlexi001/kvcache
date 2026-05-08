from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from qwen3_pyramid_kv import (
    PyramidKVConfig,
    estimate_memory_lengths,
    format_layer_block_sizes,
    patch_qwen3_pyramid_kv,
    set_trainable_scope,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_from_scratch", type=str2bool, default=False)
    parser.add_argument("--data_mode", choices=["dclm", "random_tokens"], default="dclm")
    parser.add_argument("--streaming", type=str2bool, default=True)
    parser.add_argument("--dataset_format", choices=["auto", "parquet", "json", "text"], default="auto")
    parser.add_argument("--data_files_glob", type=str, default=None)
    parser.add_argument("--random_dataset_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_length", type=int, default=4096)

    parser.add_argument("--first_full_layers", type=int, default=4)
    parser.add_argument("--last_full_layers", type=int, default=4)
    parser.add_argument("--max_block_size", type=int, default=4)
    parser.add_argument("--anchor_tokens", type=int, default=64)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--compressor_hidden_dim", type=int, default=64)
    parser.add_argument("--layer_block_sizes", type=str, default=None)
    parser.add_argument(
        "--trainable_scope",
        choices=["compressor", "compressor_and_attention", "all"],
        default="compressor",
    )

    parser.add_argument("--distill_teacher_path", type=str, default=None)
    parser.add_argument("--distill_kl_weight", type=float, default=0.0)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_last_tokens", type=int, default=0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    return parser.parse_args()


class RandomTokenDataset(TorchDataset):
    def __init__(self, size: int, seq_length: int, vocab_size: int, seed: int) -> None:
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_length,),
            generator=generator,
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class StreamingTokenBlockDataset(TorchIterableDataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        seq_length: int,
        dataset_format: str,
        data_files_glob: str | None,
    ) -> None:
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.dataset_format = dataset_format
        self.data_files_glob = data_files_glob

    def _resolve_files(self) -> tuple[str, str]:
        p = Path(self.dataset_path)
        patterns = [self.data_files_glob] if self.data_files_glob else []
        if not patterns:
            if self.dataset_format in ("auto", "parquet"):
                patterns.append("**/*.parquet")
            if self.dataset_format in ("auto", "json"):
                patterns.extend(["**/*.jsonl", "**/*.json"])
            if self.dataset_format in ("auto", "text"):
                patterns.append("**/*.txt")

        selected_format = None
        data_files = None
        for pattern in patterns:
            first_match = next(p.glob(pattern), None)
            if first_match is None:
                continue
            suffix = first_match.suffix.lower()
            if self.dataset_format != "auto":
                selected_format = self.dataset_format
            elif suffix == ".parquet":
                selected_format = "parquet"
            elif suffix in (".json", ".jsonl"):
                selected_format = "json"
            else:
                selected_format = "text"
            data_files = str(p / pattern)
            break

        if selected_format is None or data_files is None:
            raise FileNotFoundError(f"No loadable dataset files found under {self.dataset_path}")
        return selected_format, data_files

    def _iter_records(self):
        dataset_format, files = self._resolve_files()
        dataset = load_dataset(
            dataset_format,
            data_files=files,
            split="train",
            streaming=True,
        )

        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        for idx, example in enumerate(dataset):
            if idx % num_shards != shard_id:
                continue
            yield example

    def __iter__(self):
        buffer: list[int] = []
        text_col = None
        for example in self._iter_records():
            if text_col is None:
                text_col = pick_text_column_from_names(example.keys())
            text = example.get(text_col)
            if text is None:
                continue
            buffer.extend(self.tokenizer(str(text), add_special_tokens=False)["input_ids"])
            while len(buffer) >= self.seq_length:
                input_ids = torch.tensor(buffer[: self.seq_length], dtype=torch.long)
                del buffer[: self.seq_length]
                yield {"input_ids": input_ids, "labels": input_ids.clone()}


def infer_dataset_format(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in (".json", ".jsonl"):
        return "json"
    return "text"


def load_text_dataset(path: str, dataset_format: str, data_files_glob: str | None):
    p = Path(path)
    if (p / "dataset_info.json").exists() or (p / "state.json").exists():
        ds = load_from_disk(path)
    else:
        if data_files_glob:
            files = [str(x) for x in p.glob(data_files_glob)]
            if not files:
                raise FileNotFoundError(f"No files matched {data_files_glob} under {path}")
            fmt = dataset_format if dataset_format != "auto" else infer_dataset_format(files[0])
            ds = load_dataset(fmt, data_files=files, split="train")
        elif dataset_format in ("auto", "parquet") and (parquet_files := list(p.rglob("*.parquet"))):
            ds = load_dataset("parquet", data_files=[str(x) for x in parquet_files], split="train")
        elif dataset_format in ("auto", "json") and (
            json_files := list(p.rglob("*.jsonl")) + list(p.rglob("*.json"))
        ):
            ds = load_dataset("json", data_files=[str(x) for x in json_files], split="train")
        elif dataset_format in ("auto", "text") and (txt_files := list(p.rglob("*.txt"))):
            ds = load_dataset("text", data_files=[str(x) for x in txt_files], split="train")
        else:
            raise FileNotFoundError(f"No loadable dataset files found under {path}")

    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    return ds


def pick_text_column_from_names(columns) -> str:
    columns = list(columns)
    for name in ("text", "content", "document", "raw_content"):
        if name in columns:
            return name
    if columns:
        return columns[0]
    raise ValueError("Dataset has no columns")


def pick_text_column(dataset) -> str:
    return pick_text_column_from_names(dataset.column_names)


def tokenize_and_group(dataset, tokenizer, seq_length: int):
    text_col = pick_text_column(dataset)

    def tokenize(batch):
        return tokenizer(batch[text_col], add_special_tokens=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated: list[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total = len(concatenated) // seq_length * seq_length
        input_ids = [concatenated[i : i + seq_length] for i in range(0, total, seq_length)]
        return {"input_ids": input_ids, "labels": [x.copy() for x in input_ids]}

    return tokenized.map(group_texts, batched=True, desc="Grouping")


def causal_lm_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    batch = {}
    for key in ("input_ids", "labels"):
        values = [item[key] for item in features]
        tensors = [
            value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
            for value in values
        ]
        batch[key] = torch.stack(tensors)
    return batch


class DistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        distill_kl_weight: float = 0.0,
        distill_temperature: float = 2.0,
        distill_last_tokens: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_kl_weight = distill_kl_weight
        self.distill_temperature = distill_temperature
        self.distill_last_tokens = distill_last_tokens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.teacher_model is not None and self.distill_kl_weight > 0.0:
            input_device = inputs["input_ids"].device
            teacher_device = next(self.teacher_model.parameters()).device
            if teacher_device != input_device:
                self.teacher_model.to(input_device)
            teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)
            student_logits = outputs.logits
            teacher_logits = teacher_outputs.logits.to(student_logits.device)
            if self.distill_last_tokens > 0:
                student_logits = student_logits[:, -self.distill_last_tokens :, :]
                teacher_logits = teacher_logits[:, -self.distill_last_tokens :, :]
            temperature = self.distill_temperature
            kl_per_vocab = F.kl_div(
                F.log_softmax(student_logits.float() / temperature, dim=-1),
                F.softmax(teacher_logits.float() / temperature, dim=-1),
                reduction="none",
            ) * (temperature * temperature)
            kl = kl_per_vocab.sum(dim=-1).mean()
            loss = loss + self.distill_kl_weight * kl
            outputs.loss = loss
        return (loss, outputs) if return_outputs else loss


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    if args.init_from_scratch:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.to(dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    return model


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = load_model(args, dtype)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    pyramid_cfg = PyramidKVConfig(
        first_full_layers=args.first_full_layers,
        last_full_layers=args.last_full_layers,
        max_block_size=args.max_block_size,
        anchor_tokens=args.anchor_tokens,
        recent_tokens=args.recent_tokens,
        compressor_hidden_dim=args.compressor_hidden_dim,
        layer_block_sizes=args.layer_block_sizes,
    )
    block_sizes = patch_qwen3_pyramid_kv(model, pyramid_cfg)
    trainable, total = set_trainable_scope(model, args.trainable_scope)
    if args.gradient_checkpointing and args.trainable_scope != "all":
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    estimated_lengths = estimate_memory_lengths(args.seq_length, block_sizes, pyramid_cfg)

    print(f"pyramid block sizes: {format_layer_block_sizes(block_sizes)}")
    print(f"estimated memory lengths @ seq={args.seq_length}: {estimated_lengths}")
    print(f"trainable parameters: {trainable:,} / {total:,}")

    teacher_model = None
    if args.distill_teacher_path and args.distill_kl_weight > 0.0:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.distill_teacher_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        teacher_model.config.use_cache = False
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    if args.data_mode == "random_tokens":
        train_dataset = RandomTokenDataset(
            size=args.random_dataset_size,
            seq_length=args.seq_length,
            vocab_size=len(tokenizer),
            seed=args.seed,
        )
    else:
        if args.streaming:
            train_dataset = StreamingTokenBlockDataset(
                dataset_path=args.dataset_path,
                tokenizer=tokenizer,
                seq_length=args.seq_length,
                dataset_format=args.dataset_format,
                data_files_glob=args.data_files_glob,
            )
        else:
            dataset = load_text_dataset(
                args.dataset_path,
                dataset_format=args.dataset_format,
                data_files_glob=args.data_files_glob,
            )
            train_dataset = tokenize_and_group(dataset, tokenizer, args.seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=not args.bf16,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        logging_dir=str(Path(args.output_dir) / "tensorboard"),
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=causal_lm_collator,
        teacher_model=teacher_model,
        distill_kl_weight=args.distill_kl_weight,
        distill_temperature=args.distill_temperature,
        distill_last_tokens=args.distill_last_tokens,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
