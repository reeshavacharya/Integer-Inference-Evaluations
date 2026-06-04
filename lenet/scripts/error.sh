#!/usr/bin/env bash
# Error analysis script with optional filtering
# Usage: ./error.sh [--dataset DATASET] [--activation ACTIVATION] [--mode MODE]
# Valid activations: relu, gelu, leaky_relu (default: all)
# Valid modes: int8, int32, fxp32 (default: all)
# Valid datasets: MNIST, CIFAR10, Brain_MRI, OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST (default: all)
# If clamp is omitted, int32 and fxp32 run both clamped and unclamped variants.

set -euo pipefail

# Define available options
ALL_DATASETS=("CIFAR10" "Brain_MRI" "OCTMNIST" "BloodMNIST" "OrganAMNIST" "PneumoniaMNIST" "MNIST")
ALL_ACTIVATIONS=("relu" "gelu" "leaky_relu")
ALL_MODES=("int32")

# Default to all
FILTER_DATASET=""
FILTER_ACTIVATION=""
FILTER_MODE=""
# clamp removed
NUM_DATA=256
BATCH_SIZE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --activation)
            FILTER_ACTIVATION="$2"
            shift 2
            ;;
        --mode)
            FILTER_MODE="$2"
            shift 2
            ;;
        --num_data)
            NUM_DATA="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        # clamp argument removed
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--dataset DATASET] [--activation ACTIVATION] [--mode MODE] [--num_data NUM] [--batch_size B]" >&2
            exit 1
            ;;
    esac
done

# Build the actual lists to iterate
DATASETS=()
for ds in "${ALL_DATASETS[@]}"; do
    if [ -z "$FILTER_DATASET" ] || [ "$ds" = "$FILTER_DATASET" ]; then
        DATASETS+=("$ds")
    fi
done

ACTIVATIONS=()
for act in "${ALL_ACTIVATIONS[@]}"; do
    if [ -z "$FILTER_ACTIVATION" ] || [ "$act" = "$FILTER_ACTIVATION" ]; then
        ACTIVATIONS+=("$act")
    fi
done

MODES=()
for mode in "${ALL_MODES[@]}"; do
    if [ -z "$FILTER_MODE" ] || [ "$mode" = "$FILTER_MODE" ]; then
        MODES+=("$mode")
    fi
done

# Verify if specified
if [ -n "$FILTER_DATASET" ] && [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: Dataset '$FILTER_DATASET' not found." >&2
    exit 1
fi

if [ -n "$FILTER_ACTIVATION" ] && [ ${#ACTIVATIONS[@]} -eq 0 ]; then
    echo "ERROR: Activation '$FILTER_ACTIVATION' not found (must be: relu, gelu, leaky_relu)." >&2
    exit 1
fi

if [ -n "$FILTER_MODE" ] && [ ${#MODES[@]} -eq 0 ]; then
    echo "ERROR: Mode '$FILTER_MODE' not found (must be: int8, int32, fxp32)." >&2
    exit 1
fi

# clamp removed; run default modulo behavior

# Change to lenet directory
cd "$(dirname "$0")/.."

echo "[*] Error analysis starting..."
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Activations: ${ACTIVATIONS[*]}"
echo "[*] Modes: ${MODES[*]}"
echo "[*] Clamps: ${CLAMPS[*]}"

NUM_DATA_ARG=""
if [ -n "$NUM_DATA" ]; then
    NUM_DATA_ARG="--num_data $NUM_DATA"
fi

BATCH_SIZE_ARG=""
if [ -n "$BATCH_SIZE" ]; then
    BATCH_SIZE_ARG="--batch_size $BATCH_SIZE"
fi

# 1. Run error.py for all combinations
echo ""
echo "[*] Step 1: Running layer-by-layer error analysis..."
for mode in "${MODES[@]}"; do
    for act in "${ACTIVATIONS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            echo ""
            echo "[+] Error analysis: $ds ($act, $mode)"
            python3 error/error.py \
                --dataset "$ds" \
                --activation "$act" \
                --mode "$mode" \
                $NUM_DATA_ARG $BATCH_SIZE_ARG
        done
    done
done

# 2. Run error_comparison.py for all combinations
echo ""
echo "[*] Step 2: Running error comparison plots..."
for mode in "${MODES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        echo ""
        echo "[+] Error comparison: $ds ($mode)"
        python3 error/error_comparison.py \
            --dataset "$ds" \
            --mode "$mode"
    done
done

echo ""
echo "[+] Error analysis and comparison complete!"