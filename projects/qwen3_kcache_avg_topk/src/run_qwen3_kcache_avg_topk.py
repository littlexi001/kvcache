from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_kcache_avg_topk import KCacheAvgTopKConfig, patch_qwen3_kcache_avg_topk


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--prompt", default="Explain KV cache in three short sentences.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--topk_ratio", type=float, default=0.10)
    parser.add_argument("--first_sparse_layer", type=int, default=3)
    parser.add_argument("--last_sparse_layer", type=int, default=27)
    parser.add_argument("--min_blocks_to_keep", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = KCacheAvgTopKConfig(
        block_size=args.block_size,
        topk_ratio=args.topk_ratio,
        first_sparse_layer=args.first_sparse_layer,
        last_sparse_layer=args.last_sparse_layer,
        min_blocks_to_keep=args.min_blocks_to_keep,
    )
    patched_layers = patch_qwen3_kcache_avg_topk(model, cfg)
    print(f"patched sparse layers: {patched_layers}")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
