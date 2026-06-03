#!/bin/bash

# Ensure we're running from the resnet directory
cd "$(dirname "$0")/.." || exit

DATASET=""
ACTIVATION=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --activation) ACTIVATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate inputs
if [[ -z "$DATASET" || -z "$ACTIVATION" ]]; then
    echo "Error: Both --dataset and --activation are required."
    echo "Usage: ./scripts/train.sh --dataset <DATASET> --activation <ACTIVATION>"
    exit 1
fi

echo "=========================================================="
echo "Training with Dataset: $DATASET | Activation: $ACTIVATION"
echo "=========================================================="
python3 resnet18.py --train_data "$DATASET" --activation "$ACTIVATION"
echo ""
