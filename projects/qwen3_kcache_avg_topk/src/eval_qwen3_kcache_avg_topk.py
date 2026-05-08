from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_kcache_avg_topk import KCacheAvgTopKConfig, patch_qwen3_kcache_avg_topk


@dataclass
class EvalMetrics:
    name: str
    loss: float
    ppl: float
    mean_probability: float
    geometric_mean_probability: float
    tokens: int
    sequences: int
    mean_keep_block_ratio: float | None = None
    mean_keep_token_ratio: float | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--dataset_path",
        default="/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10",
    )
    parser.add_argument("--output_dir", default="projects/qwen3_kcache_avg_topk/outputs/eval")
    parser.add_argument("--max_files", type=int, default=128)
    parser.add_argument("--max_sequences", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--min_tokens", type=int, default=32)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_baseline", type=str2bool, default=True)
    parser.add_argument("--eval_sparse", type=str2bool, default=True)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--topk_ratio", type=float, default=0.10)
    parser.add_argument("--first_sparse_layer", type=int, default=3)
    parser.add_argument("--last_sparse_layer", type=int, default=27)
    parser.add_argument("--min_blocks_to_keep", type=int, default=1)
    return parser.parse_args()


def iter_text_files(dataset_path: str, max_files: int) -> Iterable[Path]:
    root = Path(dataset_path)
    if root.is_file():
        yield root
        return
    files = sorted(root.glob("*.txt"))
    if max_files > 0:
        files = files[:max_files]
    for path in files:
        yield path


def iter_token_sequences(
    tokenizer,
    dataset_path: str,
    max_files: int,
    max_sequences: int,
    seq_length: int,
    stride: int,
    min_tokens: int,
) -> Iterable[torch.Tensor]:
    produced = 0
    target_len = seq_length + 1
    for path in iter_text_files(dataset_path, max_files):
        text = path.read_text(encoding="utf-8", errors="ignore")
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < min_tokens:
            continue
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)

        if len(token_ids) <= target_len:
            yield torch.tensor(token_ids, dtype=torch.long)
            produced += 1
        else:
            for start in range(0, len(token_ids) - min_tokens + 1, stride):
                chunk = token_ids[start : start + target_len]
                if len(chunk) < min_tokens:
                    continue
                yield torch.tensor(chunk, dtype=torch.long)
                produced += 1
                if 0 < max_sequences <= produced:
                    return
        if 0 < max_sequences <= produced:
            return


def _iter_sparse_modules(model) -> list[torch.nn.Module]:
    return [
        module
        for module in model.modules()
        if hasattr(module, "_kcache_avg_topk_last_keep_blocks")
    ]


def score_sequences_decode(
    model,
    sequences: list[torch.Tensor],
    device: torch.device,
    name: str,
) -> EvalMetrics:
    total_nll = 0.0
    total_prob = 0.0
    total_tokens = 0
    keep_ratio_sum = 0.0
    keep_token_ratio_sum = 0.0
    keep_ratio_count = 0

    model.eval()
    with torch.inference_mode():
        for seq in sequences:
            if seq.numel() < 2:
                continue
            input_ids = seq.to(device).view(1, -1)
            outputs = model(input_ids=input_ids[:, :1], use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            for pos in range(1, input_ids.shape[1]):
                target = input_ids[:, pos]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                token_log_prob = log_probs.gather(-1, target.view(1, 1)).squeeze()
                total_nll += float(-token_log_prob)
                total_prob += float(token_log_prob.exp())
                total_tokens += 1

                sparse_modules = _iter_sparse_modules(model)
                if sparse_modules:
                    for module in sparse_modules:
                        num_blocks = getattr(module, "_kcache_avg_topk_last_num_blocks", 0)
                        keep_blocks = getattr(module, "_kcache_avg_topk_last_keep_blocks", 0)
                        kv_len = getattr(module, "_kcache_avg_topk_last_kv_len", 0)
                        if num_blocks > 0 and kv_len > 0:
                            keep_ratio_sum += keep_blocks / num_blocks
                            keep_token_ratio_sum += min(keep_blocks * module.kcache_avg_topk_config.block_size, kv_len) / kv_len
                            keep_ratio_count += 1

                if pos + 1 < input_ids.shape[1]:
                    outputs = model(
                        input_ids=input_ids[:, pos : pos + 1],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]

    if total_tokens == 0:
        raise ValueError("No tokens were scored. Check dataset_path/min_tokens/max_sequences.")

    loss = total_nll / total_tokens
    return EvalMetrics(
        name=name,
        loss=loss,
        ppl=math.exp(min(loss, 80.0)),
        mean_probability=total_prob / total_tokens,
        geometric_mean_probability=math.exp(-loss),
        tokens=total_tokens,
        sequences=len(sequences),
        mean_keep_block_ratio=(keep_ratio_sum / keep_ratio_count if keep_ratio_count else None),
        mean_keep_token_ratio=(keep_token_ratio_sum / keep_ratio_count if keep_ratio_count else None),
    )


def write_outputs(output_dir: Path, rows: list[EvalMetrics], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "metrics": [asdict(row) for row in rows],
    }
    if len(rows) == 2:
        base = rows[0]
        other = rows[1]
        payload["delta"] = {
            "loss": other.loss - base.loss,
            "ppl": other.ppl - base.ppl,
            "mean_probability": other.mean_probability - base.mean_probability,
            "geometric_mean_probability": (
                other.geometric_mean_probability - base.geometric_mean_probability
            ),
        }

    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sequences = list(
        iter_token_sequences(
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            max_files=args.max_files,
            max_sequences=args.max_sequences,
            seq_length=args.seq_length,
            stride=args.stride,
            min_tokens=args.min_tokens,
        )
    )
    if not sequences:
        raise ValueError(f"No usable .txt sequences found under {args.dataset_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = True

    rows: list[EvalMetrics] = []
    if args.eval_baseline:
        rows.append(score_sequences_decode(model, sequences, device, "baseline"))

    if args.eval_sparse:
        cfg = KCacheAvgTopKConfig(
            block_size=args.block_size,
            topk_ratio=args.topk_ratio,
            first_sparse_layer=args.first_sparse_layer,
            last_sparse_layer=args.last_sparse_layer,
            min_blocks_to_keep=args.min_blocks_to_keep,
        )
        patched_layers = patch_qwen3_kcache_avg_topk(model, cfg)
        print(f"patched sparse layers: {patched_layers}")
        rows.append(score_sequences_decode(model, sequences, device, "kcache_avg_topk"))

    for row in rows:
        print(
            f"{row.name}: loss={row.loss:.6f} ppl={row.ppl:.4f} "
            f"mean_prob={row.mean_probability:.8f} "
            f"geo_mean_prob={row.geometric_mean_probability:.8f} "
            f"tokens={row.tokens} sequences={row.sequences}"
        )
        if row.mean_keep_block_ratio is not None:
            print(
                f"{row.name}: mean_keep_block_ratio={row.mean_keep_block_ratio:.6f} "
                f"mean_keep_token_ratio={row.mean_keep_token_ratio:.6f}"
            )

    write_outputs(Path(args.output_dir), rows, args)
    print(f"saved metrics to {Path(args.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
