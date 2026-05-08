#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-oracle}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/lym_code/models/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-/mnt/workspace/dclm}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/qwen3_${MODE}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 40000))}"

torchrun \
  --nproc_per_node=8 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_DIR}/src/train_qwen3_chunk_sparse.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${OUT_DIR}" \
  --init_from_scratch true \
  --data_mode "${DATA_MODE:-dclm}" \
  --streaming "${STREAMING:-true}" \
  --dataset_format "${DATASET_FORMAT:-auto}" \
  --data_files_glob "${DATA_FILES_GLOB:-}" \
  --mode "${MODE}" \
  --seq_length "${SEQ_LENGTH:-1024}" \
  --num_chunks 20 \
  --keep_middle 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC:-32}" \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS:-10000}" \
  --bf16 true \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-true}" \
  --logging_steps 10 \
  --save_steps 500
