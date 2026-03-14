#!/usr/bin/env bash
# Run stt-bot server with CUDA libraries from the venv exposed to ctranslate2/faster-whisper.
# Usage: bash run.sh [uvicorn extra args]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_SITE="$SCRIPT_DIR/.venv/lib/python3.12/site-packages"

CUBLAS_LIB="$VENV_SITE/nvidia/cublas/lib"
CUDNN_LIB="$VENV_SITE/nvidia/cudnn/lib"

export LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[run.sh] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

exec "$SCRIPT_DIR/.venv/bin/uvicorn" main:app --reload --host 0.0.0.0 --port 8010 "$@"
