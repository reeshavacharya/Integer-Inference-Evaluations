"""Benchmark float vs integer inference across all LeNet datasets.

This script evaluates the same trained models used by inference.py over the
deterministic 10% test splits created in lenet5.py. It supports:
MNIST, CIFAR10, Brain-MRI, EYE, CHEST, and all Multi-Cancer submodels.

If a checkpoint for a dataset is missing, the script trains it first using
the correct train_data flag in lenet5.py, then benchmarks it.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import inference
import lenet5 as train_mod


BENCHMARK_DATASETS = [
	"MNIST",
	"CIFAR10",
	"Brain-MRI",
	"EYE",
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
		args.data_dir = "./data/MNIST/"
	elif train_data_flag == "Brain-MRI":
		args.data_dir = "./data/Brain-MRI/"
	elif train_data_flag in ("CIFR10", "CIFAR10"):
		args.data_dir = "./data/CIFAR10/"
	elif train_data_flag == "EYE":
		args.data_dir = "./data/EYE/"
	elif train_data_flag == "CHEST":
		args.data_dir = "./data/CHEST/"
	elif train_data_flag == "Multi-Cancer":
		args.data_dir = "./data/Multi-Cancer/"
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
		return setup_result[2]

	if train_mod.test_loader is not None:
		return train_mod.test_loader

	raise RuntimeError(f"Could not resolve test loader for dataset: {dataset_name}")


def _build_model(dataset_name: str) -> torch.nn.Module:
	cfg = inference._resolve_infer_config(dataset_name)
	model = cfg["model"]
	state = torch.load(cfg["model_path"], map_location="cpu")
	model.load_state_dict(state)
	model.eval()
	return model


def _float_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in loader:
			outputs = model(images)
			if labels.dim() > 1:
				preds = (torch.sigmoid(outputs) >= 0.5).float()
				correct += (preds == labels).sum().item()
				total += labels.numel()
			else:
				preds = outputs.argmax(dim=1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)

	return 100.0 * correct / max(total, 1)


def _integer_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
	"""Compute accuracy using the integer pipeline from inference.py."""
	correct = 0
	total = 0

	for images, labels in loader:
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

	return 100.0 * correct / max(total, 1)


def benchmark(dataset_names=None) -> dict:
	"""Run float and integer evaluation for all requested datasets."""
	_disable_inference_debug_trace()

	targets = dataset_names or BENCHMARK_DATASETS
	results = {}
	multi_trained = False

	for name in targets:
		print(f"\n[bench] Dataset: {name}")
		multi_trained = _ensure_checkpoint(name, multi_trained)

		loader = _get_test_loader(name)
		model = _build_model(name)
		float_acc = _float_accuracy(model, loader)
		int_acc = _integer_accuracy(model, loader)
		results[name] = {"float": float_acc, "integer": int_acc}

		print(
			f"[bench] {name}: float={float_acc:.2f}% | integer={int_acc:.2f}%"
		)

	return results


if __name__ == "__main__":
	metrics = benchmark()
	with open("benchmark_results.json", "w") as f:
		json.dump(metrics, f, indent=2)

	print("\nSaved benchmark_results.json with:")
	for ds, vals in metrics.items():
		print(f"  {ds}: float={vals['float']:.2f}%, integer={vals['integer']:.2f}%")
