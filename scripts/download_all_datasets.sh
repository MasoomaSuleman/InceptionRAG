#!/bin/bash
# InceptionRAG - Parallel Dataset Download
# Downloads NQ, HotpotQA, MSMARCO and builds Contriever embeddings

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"
LOG_DIR="$BASE_DIR/log/download"
VENV_DIR="${VENV_DIR:-$BASE_DIR/.venv}"

if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    PYTHON="$VENV_DIR/Scripts/python.exe"
elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    echo "ERROR: Could not find a Python interpreter in $VENV_DIR"
    echo "Create the venv first or set VENV_DIR to its location."
    exit 1
fi

export VENV_DIR

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "InceptionRAG Dataset Download"
echo "=============================================="
echo "Data dir: $DATA_DIR"
echo "Log dir: $LOG_DIR"
echo ""

# Download function
download_dataset() {
    local dataset=$1
    local log_file="$LOG_DIR/${dataset}_download.log"
    
    echo "[$(date +%H:%M:%S)] Starting $dataset download..."
    
    "$PYTHON" "$DATA_DIR/download_datasets.py" \
        --dataset "$dataset" \
        --output-dir "$DATA_DIR" \
        > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ $dataset completed"
    else
        echo "[$(date +%H:%M:%S)] ✗ $dataset failed, see $log_file"
    fi
}

# Run in parallel
echo "Starting parallel download of 3 datasets..."
echo ""

download_dataset "nq" &
PID_NQ=$!

download_dataset "hotpotqa" &
PID_HOTPOTQA=$!

download_dataset "msmarco" &
PID_MSMARCO=$!

# Wait for all
echo "Waiting for downloads to complete..."
echo "  NQ PID: $PID_NQ"
echo "  HotpotQA PID: $PID_HOTPOTQA"
echo "  MSMARCO PID: $PID_MSMARCO"
echo ""

wait $PID_NQ
wait $PID_HOTPOTQA
wait $PID_MSMARCO

echo ""
echo "=============================================="
echo "All downloads completed!"
echo "=============================================="
echo ""
echo "Output files:"
ls -la "$DATA_DIR/datasets/" 2>/dev/null || echo "Check datasets directory"

# Auto-run cache building
echo ""
echo "=============================================="
echo "Starting corpus cache building..."
echo "=============================================="
bash "$SCRIPT_DIR/build_all_cache.sh"
