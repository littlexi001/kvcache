#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-oracle}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${MODE}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/${MODE}.pid"

nohup bash "${SCRIPT_DIR}/run_8gpu.sh" "${MODE}" > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

echo "started ${MODE}"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "tensorboard: tensorboard --logdir ${PROJECT_DIR}/outputs/qwen3_${MODE}/tensorboard --host 0.0.0.0 --port 6006"
