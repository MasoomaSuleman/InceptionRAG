#!/bin/bash
# InceptionRAG - Build Corpus Cache for All Datasets (Parallel)
# IMPORTANT: No padding in Contriever encoding!

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"
LOG_DIR="$BASE_DIR/log/cache"
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

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "InceptionRAG Corpus Cache Builder"
echo "=============================================="
echo "Data dir: $DATA_DIR"
echo "CRITICAL: No padding in encoding!"
echo ""

# Check if datasets exist
for ds in nq hotpotqa msmarco; do
    if [ ! -d "$DATA_DIR/datasets/$ds" ]; then
        echo "ERROR: Dataset $ds not found at $DATA_DIR/datasets/$ds"
        echo "Please run download first!"
        exit 1
    fi
done

echo "All datasets found. Starting cache build..."
echo ""

# Build function
build_cache() {
    local dataset=$1
    local gpu=$2
    local log_file="$LOG_DIR/${dataset}_cache.log"
    
    echo "[$(date +%H:%M:%S)] Starting $dataset cache build (GPU $gpu)..."
    
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCRIPT_DIR/build_corpus_cache.py" \
        --dataset "$dataset" \
        --data-dir "$DATA_DIR" \
        --device cuda \
        > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ $dataset cache completed"
    else
        echo "[$(date +%H:%M:%S)] ✗ $dataset cache failed, see $log_file"
    fi
}

# Check GPU count
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -ge 3 ]; then
    # Run all 3 in parallel on different GPUs
    echo "Running 3 datasets in parallel on 3 GPUs..."
    build_cache "nq" 0 &
    build_cache "hotpotqa" 1 &
    build_cache "msmarco" 2 &
elif [ "$GPU_COUNT" -ge 1 ]; then
    # Run sequentially on single GPU
    echo "Running sequentially on single GPU..."
    build_cache "nq" 0
    build_cache "hotpotqa" 0
    build_cache "msmarco" 0
else
    echo "No GPU found, using CPU..."
    "$PYTHON" "$SCRIPT_DIR/build_corpus_cache.py" --all --data-dir "$DATA_DIR" --device cpu
    exit 0
fi

# Wait for all background jobs
wait

echo ""
echo "=============================================="
echo "Cache building completed!"
echo "=============================================="
echo ""
echo "Output files:"
ls -lh "$DATA_DIR/corpus/" 2>/dev/null || echo "No cache files yet"
