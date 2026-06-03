#!/usr/bin/env bash
# Export script for INT8/INT32 with optional filtering
# Usage: ./export.sh [--mode MODE] [--activation ACTIVATION] [--dataset DATASET]
# Valid modes: int8, int32 (default: both)
# Valid activations: relu, gelu, leaky_relu (default: all)
# Valid datasets: MNIST, CIFAR10, Brain-MRI, OCTMNIST, BloodMNIST, OrganAMNIST, PneumoniaMNIST (default: all)

set -euo pipefail

# Define available options
ALL_DATASETS=("MNIST" "CIFAR10" "Brain-MRI" "OCTMNIST" "BloodMNIST" "OrganAMNIST" "PneumoniaMNIST")
ALL_ACTIVATIONS=("relu" "gelu" "leaky_relu")
ALL_MODES=("int32")

# Default to all
FILTER_MODE=""
FILTER_ACTIVATION=""
FILTER_DATASET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            FILTER_MODE="$2"
            shift 2
            ;;
        --activation)
            FILTER_ACTIVATION="$2"
            shift 2
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--mode MODE] [--activation ACTIVATION] [--dataset DATASET]" >&2
            exit 1
            ;;
    esac
done

# Build the actual lists to iterate
MODES=()
for mode in "${ALL_MODES[@]}"; do
    if [ -z "$FILTER_MODE" ] || [ "$mode" = "$FILTER_MODE" ]; then
        MODES+=("$mode")
    fi
done

ACTIVATIONS=()
for act in "${ALL_ACTIVATIONS[@]}"; do
    if [ -z "$FILTER_ACTIVATION" ] || [ "$act" = "$FILTER_ACTIVATION" ]; then
        ACTIVATIONS+=("$act")
    fi
done

DATASETS=()
for ds in "${ALL_DATASETS[@]}"; do
    if [ -z "$FILTER_DATASET" ] || [ "$ds" = "$FILTER_DATASET" ]; then
        DATASETS+=("$ds")
    fi
done

# Verify if specified
if [ -n "$FILTER_MODE" ] && [ ${#MODES[@]} -eq 0 ]; then
    echo "ERROR: Mode '$FILTER_MODE' not found (must be: int8, int32)." >&2
    exit 1
fi

if [ -n "$FILTER_ACTIVATION" ] && [ ${#ACTIVATIONS[@]} -eq 0 ]; then
    echo "ERROR: Activation '$FILTER_ACTIVATION' not found (must be: relu, gelu, leaky_relu)." >&2
    exit 1
fi

if [ -n "$FILTER_DATASET" ] && [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: Dataset '$FILTER_DATASET' not found." >&2
    exit 1
fi

# Change to resnet directory
cd "$(dirname "$0")/.."

echo "[*] Export starting..."
echo "[*] Modes: ${MODES[*]}"
echo "[*] Activations: ${ACTIVATIONS[*]}"
echo "[*] Datasets: ${DATASETS[*]}"

# Iterate and export
for mode in "${MODES[@]}"; do
    for act in "${ACTIVATIONS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            slug=$(echo "$ds" | tr '[:upper:]' '[:lower:]' | sed 's/[ -]/_/g')
            quant_file="best_resnet18_${act}_${slug}.pth"
            
            echo ""
            echo "[+] Exporting $mode model for $ds with activation=$act"
            if [ "$mode" = "int8" ]; then
                python3 INT8/export_int8_model.py --quantize "$quant_file" --activation "$act"
            else
                python3 INT32/export_int32_model.py --quantize "$quant_file" --activation "$act"
            fi
        done
    done
done

echo ""
echo "[+] Export complete!"