#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-oracle}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/lym_code/models/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-/mnt/workspace/dclm}"
OUT_DIR="${OUT_DIR:-./outputs/qwen3_${MODE}}"

torchrun --nproc_per_node=8 src/train_qwen3_chunk_sparse.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${OUT_DIR}" \
  --init_from_scratch true \
  --data_mode "${DATA_MODE:-dclm}" \
  --mode "${MODE}" \
  --seq_length 4096 \
  --num_chunks 20 \
  --keep_middle 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --save_steps 500
