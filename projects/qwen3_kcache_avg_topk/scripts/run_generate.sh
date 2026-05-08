#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/lym_code/models/Qwen3-0.6B}"
PROMPT="${PROMPT:-Explain KV cache in three short sentences.}"

python "${PROJECT_DIR}/src/run_qwen3_kcache_avg_topk.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --bf16 "${BF16:-true}" \
  --block_size "${BLOCK_SIZE:-10}" \
  --topk_ratio "${TOPK_RATIO:-0.10}" \
  --first_sparse_layer "${FIRST_SPARSE_LAYER:-3}" \
  --last_sparse_layer "${LAST_SPARSE_LAYER:-27}" \
  --min_blocks_to_keep "${MIN_BLOCKS_TO_KEEP:-1}"
