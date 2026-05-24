#!/usr/bin/env bash
# Calibration script with optional dataset/activation filtering
# Usage: ./calibrate.sh [--dataset DATASET] [--activation ACTIVATION]
# If no args specified, calibrates all datasets with all activations.

set -euo pipefail

# Define available options
ALL_DATASETS=("MNIST" "CIFAR10" "Brain-MRI" "OCTMNIST" "BloodMNIST" "OrganAMNIST" "PneumoniaMNIST")
ALL_ACTIVATIONS=("relu" "gelu" "leaky_relu")

# Default to all
FILTER_DATASET=""
FILTER_ACTIVATION=""

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
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--dataset DATASET] [--activation ACTIVATION]" >&2
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

# Verify dataset/activation are valid if specified
if [ -n "$FILTER_DATASET" ] && [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: Dataset '$FILTER_DATASET' not found." >&2
    exit 1
fi

if [ -n "$FILTER_ACTIVATION" ] && [ ${#ACTIVATIONS[@]} -eq 0 ]; then
    echo "ERROR: Activation '$FILTER_ACTIVATION' not found (must be: relu, gelu, leaky_relu)." >&2
    exit 1
fi

# Change to resnet directory (assumes script is in resnet/scripts/)
cd "$(dirname "$0")/.."

echo "[*] Calibration starting..."
echo "[*] Datasets: ${DATASETS[*]}"
echo "[*] Activations: ${ACTIVATIONS[*]}"

# Iterate and calibrate
for ds in "${DATASETS[@]}"; do
    for act in "${ACTIVATIONS[@]}"; do
        echo ""
        echo "[+] Calibrating $ds with activation=$act"
        python3 calibration.py --dataset "$ds" --activation "$act"
    done
done

echo ""
echo "[+] Calibration complete!"
