#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-compressor}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/lym_code/models/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-/mnt/workspace/dclm}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 40000))}"

case "${STAGE}" in
  sanity)
    DATA_MODE="${DATA_MODE:-random_tokens}"
    MAX_STEPS="${MAX_STEPS:-5}"
    TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-compressor}"
    LR="${LR:-2e-4}"
    OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/sanity}"
    ;;
  compressor)
    DATA_MODE="${DATA_MODE:-dclm}"
    MAX_STEPS="${MAX_STEPS:-10000}"
    TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-compressor}"
    LR="${LR:-2e-4}"
    OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/compressor}"
    ;;
  attention)
    DATA_MODE="${DATA_MODE:-dclm}"
    MAX_STEPS="${MAX_STEPS:-10000}"
    TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-compressor_and_attention}"
    LR="${LR:-5e-5}"
    OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/attention}"
    ;;
  full)
    DATA_MODE="${DATA_MODE:-dclm}"
    MAX_STEPS="${MAX_STEPS:-10000}"
    TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-all}"
    LR="${LR:-1e-5}"
    OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/full}"
    ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    echo "valid stages: sanity, compressor, attention, full" >&2
    exit 2
    ;;
esac

ARGS=(
  --model_name_or_path "${MODEL_PATH}"
  --dataset_path "${DATA_PATH}"
  --output_dir "${OUT_DIR}"
  --init_from_scratch false
  --data_mode "${DATA_MODE}"
  --streaming "${STREAMING:-true}"
  --dataset_format "${DATASET_FORMAT:-auto}"
  --data_files_glob "${DATA_FILES_GLOB:-}"
  --seq_length "${SEQ_LENGTH}"
  --first_full_layers "${FIRST_FULL_LAYERS:-4}"
  --last_full_layers "${LAST_FULL_LAYERS:-4}"
  --max_block_size "${MAX_BLOCK_SIZE:-4}"
  --anchor_tokens "${ANCHOR_TOKENS:-64}"
  --recent_tokens "${RECENT_TOKENS:-512}"
  --compressor_hidden_dim "${COMPRESSOR_HIDDEN_DIM:-64}"
  --trainable_scope "${TRAINABLE_SCOPE}"
  --distill_teacher_path "${DISTILL_TEACHER_PATH:-}"
  --distill_kl_weight "${DISTILL_KL_WEIGHT:-0.0}"
  --distill_temperature "${DISTILL_TEMPERATURE:-2.0}"
  --distill_last_tokens "${DISTILL_LAST_TOKENS:-0}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --learning_rate "${LR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --max_steps "${MAX_STEPS}"
  --warmup_steps "${WARMUP_STEPS:-100}"
  --bf16 "${BF16:-true}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --save_steps "${SAVE_STEPS:-500}"
)

if [[ -n "${LAYER_BLOCK_SIZES:-}" ]]; then
  ARGS+=(--layer_block_sizes "${LAYER_BLOCK_SIZES}")
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_DIR}/src/train_qwen3_pyramid_kv.py" \
  "${ARGS[@]}"
