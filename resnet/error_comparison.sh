#!/usr/bin/env bash
# Dynamic calibration / export / error-run script that requires an activation
# Usage: ./calibrate_export_error.sh <activation>

set -euo pipefail

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 <activation>" >&2
	echo "Example: $0 leaky_relu" >&2
	exit 2
fi

ACTIVATION="$1"

DATASETS=("MNIST" "CIFAR10" "Brain-MRI" "OCTMNIST" "BloodMNIST" "OrganAMNIST" "PneumoniaMNIST")

for ds in "${DATASETS[@]}"; do
	# slug: lowercase, replace spaces and hyphens with underscore
	slug=$(echo "$ds" | tr '[:upper:]' '[:lower:]' | sed 's/[ -]/_/g')

	echo "[+] Running error analysis for $ds"
	python3 error/error.py --dataset "$ds" --num_data 256 --activation "$ACTIVATION" --mode int8
    python3 error/error.py --dataset "$ds" --num_data 256 --activation "$ACTIVATION" --mode int32

	echo "[+] Running comparison plot for $ds"
	python3 error/error_comparison.py --dataset "$ds" --mode int8
    python3 error/error_comparison.py --dataset "$ds" --mode int32
done