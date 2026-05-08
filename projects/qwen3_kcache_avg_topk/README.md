# Qwen3 K-cache Average Top-k Inference

This project patches Qwen3-0.6B inference so that layers 0-2 keep the original
attention path, while layers 3-27 use averaged K-cache block selection during
decode.

For patched layers:

1. Split the current K cache into consecutive blocks of 10 tokens.
2. Average the keys inside each block.
3. Compute attention scores between the current query and the averaged keys.
4. Keep the top 10% highest-scoring blocks.
5. Run exact attention with the original K cache and V cache tokens belonging to
   those selected blocks.

The prefill pass uses full attention by default so the normal KV cache is built
without approximation. The sparse selection is applied to decode steps where the
query length is 1.

## Run Generation

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
bash projects/qwen3_kcache_avg_topk/scripts/run_generate.sh
```

Useful overrides:

```bash
PROMPT="Explain long-context inference." \
MAX_NEW_TOKENS=256 \
BLOCK_SIZE=10 \
TOPK_RATIO=0.10 \
FIRST_SPARSE_LAYER=3 \
LAST_SPARSE_LAYER=27 \
bash projects/qwen3_kcache_avg_topk/scripts/run_generate.sh
```

## Evaluate PPL, Loss, and Probability

The evaluation script compares normal Qwen3 decoding with this sparse decode
method on `.txt` files. It scores tokens one by one with `use_cache=True`, so the
patched decode path is actually exercised.

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
DATA_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10 \
bash projects/qwen3_kcache_avg_topk/scripts/run_eval.sh
```

Outputs are written to:

```text
projects/qwen3_kcache_avg_topk/outputs/eval/metrics.json
projects/qwen3_kcache_avg_topk/outputs/eval/metrics.csv
```

Useful quick-test overrides:

```bash
MAX_FILES=4 MAX_SEQUENCES=8 SEQ_LENGTH=256 \
bash projects/qwen3_kcache_avg_topk/scripts/run_eval.sh
```

Run evaluation in the background with `nohup`:

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
DATA_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10 \
bash projects/qwen3_kcache_avg_topk/scripts/nohup_run_eval.sh
```

Check the process and watch the log:

```bash
cat projects/qwen3_kcache_avg_topk/logs/eval.pid
tail -f projects/qwen3_kcache_avg_topk/logs/eval_*.log
```

## Notes

- This is an inference experiment and does not train new parameters.
- The implementation keeps the original KV cache intact; it only changes which
  cached tokens participate in attention for patched decode layers.
- The exact attention stage currently masks unselected original tokens. It is
  behaviorally faithful to the block-selection idea, but it is not yet a
  low-level optimized gather kernel.
- Main advantage: decode attention can inspect far fewer original K/V tokens in
  layers 3-27 when paired with a real gather/kernel implementation.
- Main risk: averaged keys can hide a single important token inside a block, so
  top-k block selection may drop useful original K/V tokens and increase PPL.
