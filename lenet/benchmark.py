"""Benchmark float vs integer inference across all LeNet datasets.

This script evaluates the same trained models used by inference.py over the
deterministic 10% test splits created in lenet5.py. It supports:
MNIST, CIFAR10, Brain-MRI, CHEST, and all Multi-Cancer submodels.

If a checkpoint for a dataset is missing, the script trains it first using
the correct train_data flag in lenet5.py, then benchmarks it.
"""

import argparse
import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

import inference
import lenet5 as train_mod


BENCHMARK_DATASETS = [
	"MNIST",
	"CIFAR10",
	"Brain-MRI",
	"CHEST",
	"Brain-Cancer",
	"Breast-Cancer",
	"Cervical-Cancer",
	"Kidney-Cancer",
	"Lung-And-Colon-Cancer",
	"Lymphoma-Cancer",
	"Oral-Cancer",
]

MULTI_CANCER_DATASETS = {
	"Brain-Cancer",
	"Breast-Cancer",
	"Cervical-Cancer",
	"Kidney-Cancer",
	"Lung-And-Colon-Cancer",
	"Lymphoma-Cancer",
	"Oral-Cancer",
}


def _disable_inference_debug_trace():
	"""Turn off heavy tensor logging while benchmarking full test loaders."""

	class _NoOpList(list):
		def append(self, item):  # type: ignore[override]
			return None

	inference.debug_trace = {  # type: ignore[attr-defined]
		"input": None,
		"layers": _NoOpList(),
		"pooling": _NoOpList(),
	}


def _train_dataset_for_checkpoint(dataset_name: str, multi_trained: bool) -> bool:
	"""Train missing checkpoint with the correct lenet5 train_data selection.

	Returns updated multi_trained flag.
	"""

	if dataset_name in MULTI_CANCER_DATASETS:
		if multi_trained:
			return multi_trained
		train_data_flag = "Multi-Cancer"
	else:
		train_data_flag = dataset_name

	print(f"[train] Missing checkpoint for {dataset_name}. Training with --train_data {train_data_flag}.")

	args = argparse.Namespace(
		batch_size=64,
		learning_rate=1e-3,
		train_data=train_data_flag,
		data_dir="",
	)

	if train_data_flag == "MNIST":
		args.data_dir = train_mod.DATA_MNIST_DIR
	elif train_data_flag == "Brain-MRI":
		args.data_dir = train_mod.DATA_BRAIN_MRI_DIR
	elif train_data_flag in ("CIFR10", "CIFAR10"):
		args.data_dir = train_mod.DATA_CIFAR10_DIR
	elif train_data_flag == "CHEST":
		args.data_dir = train_mod.DATA_CHEST_DIR
	elif train_data_flag == "Multi-Cancer":
		args.data_dir = train_mod.DATA_MULTI_CANCER_DIR
	else:
		raise ValueError(f"Unsupported train_data flag: {train_data_flag}")

	train_mod.datasetDownloader(train_data_flag)
	train_mod.main(args)

	if train_data_flag == "Multi-Cancer":
		return True
	return multi_trained


def _ensure_checkpoint(dataset_name: str, multi_trained: bool) -> bool:
	cfg = inference._resolve_infer_config(dataset_name)
	model_path = cfg["model_path"]

	if os.path.exists(model_path):
		return multi_trained

	multi_trained = _train_dataset_for_checkpoint(dataset_name, multi_trained)

	if not os.path.exists(model_path):
		raise RuntimeError(
			f"Checkpoint still missing after training for {dataset_name}: {model_path}"
		)
	return multi_trained


def _get_test_loader(dataset_name: str, batch_size: int = 64) -> DataLoader:
	"""Return test loader from the same deterministic 80/10/10 split logic."""

	cfg = inference._resolve_infer_config(dataset_name)
	# Prevent cross-dataset leakage of stale global loaders.
	train_mod.train_loader = None
	train_mod.val_loader = None
	train_mod.test_loader = None
	setup_result = cfg["setup_fn"](batch_size=batch_size)

	# Prefer explicit loader tuples returned by setup helpers
	# (used by per-cancer Multi-Cancer setup functions).
	if (
		isinstance(setup_result, tuple)
		and len(setup_result) >= 3
		and hasattr(setup_result[2], "dataset")
	):
		loader = setup_result[2]
		train_mod.validate_loader_preprocessing(loader, dataset_name, stage="benchmark")
		return loader

	if train_mod.test_loader is not None:
		train_mod.validate_loader_preprocessing(
			train_mod.test_loader, dataset_name, stage="benchmark"
		)
		return train_mod.test_loader

	raise RuntimeError(f"Could not resolve test loader for dataset: {dataset_name}")


def _build_model(dataset_name: str) -> torch.nn.Module:
	cfg = inference._resolve_infer_config(dataset_name)
	model = cfg["model"]
	state = torch.load(cfg["model_path"], map_location="cpu")
	model.load_state_dict(state)
	model.eval()
	return model


def _float_accuracy(model: torch.nn.Module, loader: DataLoader, dataset_name: str = "dataset", num_data: Optional[int] = None) -> float:
	total_samples = len(loader.dataset)
	target_samples = total_samples if num_data is None else min(num_data, total_samples)
	print(f"[bench][{dataset_name}][float] Starting benchmark over {target_samples}/{total_samples} samples.")
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (images, labels) in enumerate(loader, 1):
			if num_data is not None and total >= num_data:
				break
			
			if num_data is not None and total + images.size(0) > num_data:
				keep = num_data - total
				images = images[:keep]
				labels = labels[:keep]
			
			outputs = model(images)
			if labels.dim() > 1:
				preds = (torch.sigmoid(outputs) >= 0.5).float()
				correct += (preds == labels).sum().item()
				total += labels.numel()
			else:
				preds = outputs.argmax(dim=1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)
		
			remaining = max(target_samples - total, 0)
			print(f"[bench][{dataset_name}][float] Batch {batch_idx}: processed {total}/{target_samples} samples, remaining {remaining}.")

	return 100.0 * correct / max(total, 1)


def _integer_accuracy(model: torch.nn.Module, loader: DataLoader, dataset_name: str = "dataset", num_data: Optional[int] = None) -> float:
	"""Compute accuracy using the integer pipeline from inference.py."""
	total_samples = len(loader.dataset)
	target_samples = total_samples if num_data is None else min(num_data, total_samples)
	print(f"[bench][{dataset_name}][integer] Starting benchmark over {target_samples}/{total_samples} samples.")
	correct = 0
	total = 0

	for batch_idx, (images, labels) in enumerate(loader, 1):
		if num_data is not None and total >= num_data:
			break
		
		if num_data is not None and total + images.size(0) > num_data:
			keep = num_data - total
			images = images[:keep]
			labels = labels[:keep]
		
		# Calibration on current batch
		inference.activation_ranges.clear()
		handles = inference.register_hooks(model)
		with torch.no_grad():
			_ = model(images)
		for h in handles:
			h.remove()

		# Quantize input
		in_range = inference.activation_ranges["conv1"]
		pseudo_in_tensor = torch.tensor(
			[in_range["in_min"], in_range["in_max"]], dtype=torch.float32
		)
		scale_in, zp_in = inference.get_quantization_params(pseudo_in_tensor, num_bits=8)
		q_x = inference.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

		# Integer forward (mirrors inference.py)
		cfg = inference._get_layer_config(model)

		q_x, s_out, z_out, _, _, _ = inference.run_integer_layer(
			q_x,
			cfg["conv1"],
			"conv1",
			scale_in,
			zp_in,
			apply_relu=True,
			is_conv=True,
		)
		q_x = inference.avg_pool_uint8(q_x, name="bench_pool_after_conv1")

		q_x, s_out, z_out, _, _, _ = inference.run_integer_layer(
			q_x,
			cfg["conv2"],
			"conv2",
			s_out,
			z_out,
			apply_relu=True,
			is_conv=True,
		)
		q_x = inference.avg_pool_uint8(q_x, name="bench_pool_after_conv2")

		q_x = q_x.view(q_x.size(0), -1)

		q_x, s_out, z_out, _, _, _ = inference.run_integer_layer(
			q_x,
			cfg["fc1"],
			"fc1",
			s_out,
			z_out,
			apply_relu=True,
			is_conv=False,
		)
		q_x, s_out, z_out, _, _, _ = inference.run_integer_layer(
			q_x,
			cfg["fc2"],
			"fc2",
			s_out,
			z_out,
			apply_relu=True,
			is_conv=False,
		)

		q_out, final_s, final_z, _, _, _ = inference.run_integer_layer(
			q_x,
			cfg["fc3"],
			"fc3",
			s_out,
			z_out,
			apply_relu=False,
			is_conv=False,
		)

		int_logits = q_out.to(torch.float32)
		dequantized_logits = final_s * (int_logits - final_z)

		if labels.dim() > 1:
			int_preds = (torch.sigmoid(dequantized_logits) >= 0.5).float()
			correct += (int_preds == labels).sum().item()
			total += labels.numel()
		else:
			int_preds = dequantized_logits.argmax(dim=1)
			correct += (int_preds == labels).sum().item()
			total += labels.size(0)
		
		remaining = max(target_samples - total, 0)
		print(f"[bench][{dataset_name}][integer] Batch {batch_idx}: processed {total}/{target_samples} samples, remaining {remaining}.")

	return 100.0 * correct / max(total, 1)


def benchmark(dataset_names=None, num_data: Optional[int] = None) -> dict:
	"""Run float and integer evaluation for all requested datasets.
	If num_data is provided, only the first num_data samples from each
	test split are used for a quicker benchmark.
	"""
	_disable_inference_debug_trace()

	targets = dataset_names or BENCHMARK_DATASETS
	results = {}
	multi_trained = False

	for name in targets:
		print(f"\n[bench] Dataset: {name}")
		multi_trained = _ensure_checkpoint(name, multi_trained)

		loader = _get_test_loader(name)
		model = _build_model(name)
		float_acc = _float_accuracy(model, loader, name, num_data=num_data)
		int_acc = _integer_accuracy(model, loader, name, num_data=num_data)
		results[name] = {"float": float_acc, "integer": int_acc}

		print(
			f"[bench] {name}: float={float_acc:.2f}% | integer={int_acc:.2f}%"
		)

	return results

def _get_results_filename(single_dataset_name: Optional[str]) -> str:
	"""Build output filename for benchmark results."""
	if single_dataset_name is None:
		return "benchmark_results.json"

	suffix = single_dataset_name.lower().replace("-", "_")
	return f"benchmark_results_{suffix}.json"

def _normalize_bench_name(name: str) -> str:
	"""Normalize CLI dataset names to benchmark dataset identifiers."""
	name_upper = name.upper()
	if name_upper == "CIFR10":
		return "CIFAR10"
	if name_upper == "BRAIN-MRI":
		return "Brain-MRI"
	if name_upper == "BRAIN-CANCER":
		return "Brain-Cancer"
	if name_upper == "BREAST-CANCER":
		return "Breast-Cancer"
	if name_upper == "CERVICAL-CANCER":
		return "Cervical-Cancer"
	if name_upper == "KIDNEY-CANCER":
		return "Kidney-Cancer"
	if name_upper == "LUNG-AND-COLON-CANCER":
		return "Lung-And-Colon-Cancer"
	if name_upper == "LYMPHOMA-CANCER":
		return "Lymphoma-Cancer"
	if name_upper == "ORAL-CANCER":
		return "Oral-Cancer"
	if name_upper == "MNIST":
		return "MNIST"
	if name_upper == "CIFAR10":
		return "CIFAR10"
	if name_upper == "CHEST":
		return "CHEST"
	raise ValueError(f"Unknown benchmark dataset: {name}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--bench",
		type=str,
		default=None,
		help=(
			"Benchmark a single dataset: MNIST, CIFAR10, Brain-MRI, CHEST, "
			"Brain-Cancer, Breast-Cancer, Cervical-Cancer, Kidney-Cancer, "
			"Lung-And-Colon-Cancer, Lymphoma-Cancer, Oral-Cancer"
		),
	)
	parser.add_argument(
		"--num_data",
		type=int,
		default=None,
		help="Number of test samples to benchmark per dataset. If omitted, benchmarks the full test split.",
	)
	args = parser.parse_args()

	if args.bench is None:
		targets = None
		single_name = None
	else:
		single_name = _normalize_bench_name(args.bench)
		targets = [single_name]

	metrics = benchmark(dataset_names=targets, num_data=args.num_data)
	results_file = _get_results_filename(single_name)
	with open(results_file, "w") as f:
		json.dump(metrics, f, indent=2)

	print(f"\nSaved {results_file} with:")
	for ds, vals in metrics.items():
		print(f"  {ds}: float={vals['float']:.2f}%, integer={vals['integer']:.2f}%")
