#!/usr/bin/env bash
# Benchmark script supporting multiple modes, datasets, activations
# Usage: ./benchmark.sh [--bench DATASET] [--num_data N] [--mode MODE] [--activation ACTIVATION]

set -euo pipefail

ALL_DATASETS=("CIFAR10" "Brain-MRI" "OCTMNIST" "OrganAMNIST" "BloodMNIST" "PneumoniaMNIST" "MNIST")
ALL_MODES=("fp32" "int8" "int32" "fxp32" "fxp64")
ALL_ACTIVATIONS=("relu" "gelu" "leaky_relu")

# Defaults: run all
FILTER_DATASET=""
FILTER_MODE=""
FILTER_ACTIVATION=""
FILTER_CLAMP=""
NUM_DATA=""
BATCH_SIZE=""

usage() {
	echo "Usage: $0 [--bench DATASET] [--num_data N] [--batch_size B] [--mode MODE] [--activation ACTIVATION]"
	echo "  --bench DATASET       : single dataset to benchmark (default: all)"
	echo "  --num_data N          : number of test images to use (default: full test split)"
	echo "  --batch_size B        : batch size for inference (default: 64)"
	echo "  --mode MODE           : one mode from ${ALL_MODES[*]} (default: all)"
	echo "  --activation ACT      : one activation from ${ALL_ACTIVATIONS[*]} (default: all)"
	# clamp argument removed
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--bench)
			FILTER_DATASET="$2"
			shift 2
			;;
		--num_data)
			NUM_DATA="$2"
			shift 2
			;;
		--mode)
			FILTER_MODE="$2"
			shift 2
			;;
		--activation)
			FILTER_ACTIVATION="$2"
			shift 2
			;;
		--batch_size)
			BATCH_SIZE="$2"
			shift 2
			;;
		# clamp argument removed
		-h|--help)
			usage
			;;
		*)
			echo "Unknown option: $1" >&2
			usage
			;;
	esac
done

# Build lists
DATASETS=()
for ds in "${ALL_DATASETS[@]}"; do
	if [ -z "$FILTER_DATASET" ] || [ "$ds" = "$FILTER_DATASET" ]; then
		DATASETS+=("$ds")
	fi
done

MODES=()
for m in "${ALL_MODES[@]}"; do
	if [ -z "$FILTER_MODE" ] || [ "$m" = "$FILTER_MODE" ]; then
		MODES+=("$m")
	fi
done

ACTIVATIONS=()
for a in "${ALL_ACTIVATIONS[@]}"; do
	if [ -z "$FILTER_ACTIVATION" ] || [ "$a" = "$FILTER_ACTIVATION" ]; then
		ACTIVATIONS+=("$a")
	fi
done

if [ -n "$FILTER_DATASET" ] && [ ${#DATASETS[@]} -eq 0 ]; then
	echo "ERROR: dataset '$FILTER_DATASET' not recognized" >&2
	exit 1
fi

if [ -n "$FILTER_MODE" ] && [ ${#MODES[@]} -eq 0 ]; then
	echo "ERROR: mode '$FILTER_MODE' not recognized" >&2
	exit 1
fi

if [ -n "$FILTER_ACTIVATION" ] && [ ${#ACTIVATIONS[@]} -eq 0 ]; then
	echo "ERROR: activation '$FILTER_ACTIVATION' not recognized" >&2
	exit 1
fi

# clamp argument removed; run default behavior (unclamped modulo math)

cd "$(dirname "$0")/.."

NUM_DATA_ARG=""
if [ -n "$NUM_DATA" ]; then
	NUM_DATA_ARG="--num_data $NUM_DATA"
fi

BATCH_SIZE_ARG=""
if [ -n "$BATCH_SIZE" ]; then
	BATCH_SIZE_ARG="--batch_size $BATCH_SIZE"
fi

echo "[*] Benchmark: datasets=${DATASETS[*]} modes=${MODES[*]} activations=${ACTIVATIONS[*]} clamps=${CLAMPS[*]} num_data=${NUM_DATA:-full} batch_size=${BATCH_SIZE:-64}"

# Runner mapping for modes
run_mode() {
	local mode="$1"
	local ds="$2"
	local act="$3"
	case "$mode" in
		fp32)
			python3 benchmark.py --bench "$ds" --activation "$act" ${NUM_DATA_ARG} ${BATCH_SIZE_ARG} --mode fp32
			;;
		int8)
			python3 benchmark.py --bench "$ds" --activation "$act" ${NUM_DATA_ARG} ${BATCH_SIZE_ARG} --mode int8
			;;
		int32)
			python3 benchmark.py --bench "$ds" --activation "$act" ${NUM_DATA_ARG} ${BATCH_SIZE_ARG} --mode int32
			;;
		fxp32)
			python3 benchmark.py --bench "$ds" --activation "$act" ${NUM_DATA_ARG} ${BATCH_SIZE_ARG} --mode fxp32
			;;
		fxp64)
			python3 benchmark.py --bench "$ds" --activation "$act" ${NUM_DATA_ARG} ${BATCH_SIZE_ARG} --mode fxp64
			;;
		*)
			echo "Unknown mode: $mode" >&2
			return 1
			;;
	esac
}

for ds in "${DATASETS[@]}"; do
	for act in "${ACTIVATIONS[@]}"; do
		for m in "${MODES[@]}"; do
			echo ""
			echo "[+] Benchmarking: $ds (activation=$act, mode=$m)"
			run_mode "$m" "$ds" "$act"
		done
	done
done

echo "[+] Benchmarking complete"
