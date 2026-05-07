#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-compressor}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/${STAGE}.pid"

nohup bash "${SCRIPT_DIR}/run_8gpu.sh" "${STAGE}" > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

echo "started ${STAGE}"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "tensorboard: tensorboard --logdir ${PROJECT_DIR}/outputs --host 0.0.0.0 --port 6006"
