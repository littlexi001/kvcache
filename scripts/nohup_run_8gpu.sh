#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-oracle}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${MODE}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/${MODE}.pid"

nohup bash scripts/run_8gpu.sh "${MODE}" > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

echo "started ${MODE}"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "tensorboard: tensorboard --logdir ./outputs/qwen3_${MODE}/tensorboard --host 0.0.0.0 --port 6006"
