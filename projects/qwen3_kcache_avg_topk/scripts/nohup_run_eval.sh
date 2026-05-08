#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

JOB_NAME="${JOB_NAME:-eval}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/${JOB_NAME}.pid"

nohup bash "${SCRIPT_DIR}/run_eval.sh" > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

echo "started ${JOB_NAME}"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "watch: tail -f ${LOG_FILE}"
echo "metrics: ${OUT_DIR:-${PROJECT_DIR}/outputs/eval}/metrics.json"
