from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from qwen3_chunk_attention import ChunkSparseConfig, patch_qwen3_chunk_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_from_scratch", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--data_mode", choices=["dclm", "random_tokens"], default="dclm")
    parser.add_argument("--random_dataset_size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["baseline", "oracle", "router"], default="oracle")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--num_chunks", type=int, default=20)
    parser.add_argument("--keep_middle", type=int, default=3)
    parser.add_argument("--router_dim", type=int, default=128)
    parser.add_argument("--router_aux_weight", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=True)
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


def load_text_dataset(path: str):
    p = Path(path)
    if (p / "dataset_info.json").exists() or (p / "state.json").exists():
        ds = load_from_disk(path)
    else:
        parquet_files = list(p.rglob("*.parquet"))
        json_files = list(p.rglob("*.jsonl")) + list(p.rglob("*.json"))
        txt_files = list(p.rglob("*.txt"))
        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(x) for x in parquet_files], split="train")
        elif json_files:
            ds = load_dataset("json", data_files=[str(x) for x in json_files], split="train")
        elif txt_files:
            ds = load_dataset("text", data_files=[str(x) for x in txt_files], split="train")
        else:
            raise FileNotFoundError(f"No loadable dataset files found under {path}")

    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    return ds


def pick_text_column(dataset) -> str:
    columns = dataset.column_names
    for name in ["text", "content", "document", "raw_content"]:
        if name in columns:
            return name
    for name in columns:
        return name
    raise ValueError("Dataset has no columns")


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
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total = len(concatenated) // seq_length * seq_length
        input_ids = [
            concatenated[i : i + seq_length]
            for i in range(0, total, seq_length)
        ]
        return {"input_ids": input_ids, "labels": [x.copy() for x in input_ids]}

    return tokenized.map(group_texts, batched=True, desc="Grouping")


def causal_lm_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    batch = {}
    for key in ["input_ids", "labels"]:
        values = [item[key] for item in features]
        values = [
            value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
            for value in values
        ]
        batch[key] = torch.stack(values)
    return batch


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
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

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    sparse_cfg = ChunkSparseConfig(
        mode=args.mode,
        num_chunks=args.num_chunks,
        keep_middle=args.keep_middle,
        router_dim=args.router_dim,
        router_aux_weight=args.router_aux_weight,
    )
    patch_qwen3_chunk_attention(model, sparse_cfg)

    if args.data_mode == "random_tokens":
        train_dataset = RandomTokenDataset(
            size=args.random_dataset_size,
            seq_length=args.seq_length,
            vocab_size=len(tokenizer),
            seed=args.seed,
        )
    else:
        dataset = load_text_dataset(args.dataset_path)
        train_dataset = tokenize_and_group(dataset, tokenizer, args.seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=causal_lm_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
