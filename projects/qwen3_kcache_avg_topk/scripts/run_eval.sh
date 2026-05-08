#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/lym_code/models/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/eval}"

python "${PROJECT_DIR}/src/eval_qwen3_kcache_avg_topk.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${OUT_DIR}" \
  --max_files "${MAX_FILES:-128}" \
  --max_sequences "${MAX_SEQUENCES:-128}" \
  --seq_length "${SEQ_LENGTH:-1024}" \
  --stride "${STRIDE:-1024}" \
  --min_tokens "${MIN_TOKENS:-32}" \
  --bf16 "${BF16:-true}" \
  --device "${DEVICE:-cuda}" \
  --eval_baseline "${EVAL_BASELINE:-true}" \
  --eval_sparse "${EVAL_SPARSE:-true}" \
  --block_size "${BLOCK_SIZE:-10}" \
  --topk_ratio "${TOPK_RATIO:-0.10}" \
  --first_sparse_layer "${FIRST_SPARSE_LAYER:-3}" \
  --last_sparse_layer "${LAST_SPARSE_LAYER:-27}" \
  --min_blocks_to_keep "${MIN_BLOCKS_TO_KEEP:-1}"
