"""Benchmark float vs integer inference on MNIST, CIFAR10, and Brain-MRI.

Runs the same models and integer pipeline as inference.py over the
corresponding 10% test partitions defined in lenet5.py and saves
accuracies to benchmark_results.json in the form:

{
	"MNIST": {"float": acc_float, "integer": acc_int},
	"CIFAR10": {"float": acc_float, "integer": acc_int},
	"Brain-MRI": {"float": acc_float, "integer": acc_int}
}
"""

import json

import torch
from torch.utils.data import DataLoader

import inference
import lenet5 as train_mod


def _disable_inference_debug_trace():
	"""Turn off heavy debug logging in inference when benchmarking.

	This prevents run_integer_layer/pool_uint8 from storing full tensors
	in inference.debug_trace while we iterate over entire datasets.
	"""

	class _NoOpList(list):
		def append(self, item):  # type: ignore[override]
			# discard logged items to keep memory usage low
			return None

	inference.debug_trace = {  # type: ignore[attr-defined]
		"input": None,
		"layers": _NoOpList(),
		"pooling": _NoOpList(),
	}


def _get_test_loader(dataset_name: str, batch_size: int = 64) -> DataLoader:
	"""Return the test DataLoader using lenet5.py's 80/10/10 split.

	This ensures benchmarking uses the same 10% test partition as training
	for MNIST, CIFAR10, and Brain-MRI.
	"""

	name = dataset_name.upper()
	if name == "MNIST":
		train_mod.setup_MNIST(batch_size)
	elif name == "BRAIN-MRI":
		train_mod.setup_Brain_MRI(batch_size)
	elif name in ("CIFR10", "CIFAR10"):
		train_mod.setup_CIFAR10(batch_size)
	else:
		raise ValueError(f"Unknown dataset: {dataset_name}")

	return train_mod.test_loader


def _build_model(dataset_name: str) -> torch.nn.Module:
	"""Construct the trained model corresponding to the dataset."""
	name = dataset_name.upper()
	if name == "MNIST":
		model = inference.LeNet5(num_classes=10, in_channels=1)
		model_path = "best_lenet5_mnist.pth"
	elif name == "BRAIN-MRI":
		model = inference.BrainMRILeNet(num_classes=4)
		model_path = "best_lenet5_brain_mri.pth"
	elif name in ("CIFR10", "CIFAR10"):
		model = inference.LeNet5(num_classes=10, in_channels=3)
		model_path = "best_lenet5_cifar10.pth"
	else:
		raise ValueError(f"Unknown dataset: {dataset_name}")

	state = torch.load(model_path, map_location="cpu")
	model.load_state_dict(state)
	model.eval()
	return model


def _float_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			outputs = model(images)
			preds = outputs.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	return 100.0 * correct / max(total, 1)


def _integer_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
	"""Compute accuracy using the integer pipeline from inference.py."""
	correct = 0
	total = 0

	for images, labels in loader:
		# 1) Calibration for this batch
		inference.activation_ranges.clear()
		handles = inference.register_hooks(model)
		with torch.no_grad():
			_ = model(images)
		for h in handles:
			h.remove()

		# 2) Quantize input using conv1 input range
		in_range = inference.activation_ranges["conv1"]
		pseudo_in_tensor = torch.tensor(
			[in_range["in_min"], in_range["in_max"]], dtype=torch.float32
		)
		scale_in, zp_in = inference.get_quantization_params(
			pseudo_in_tensor, num_bits=8
		)
		q_x = inference.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

		# 3) Integer forward pass (mirrors inference.main)
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
		q_x = inference.pool_uint8(q_x)

		q_x, s_out, z_out, _, _, _ = inference.run_integer_layer(
			q_x,
			cfg["conv2"],
			"conv2",
			s_out,
			z_out,
			apply_relu=True,
			is_conv=True,
		)
		q_x = inference.pool_uint8(q_x)

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

		# 4) Dequantize logits and compute predictions
		int_logits = q_out.to(torch.float32)
		dequantized_logits = final_s * (int_logits - final_z)
		int_preds = dequantized_logits.argmax(dim=1)

		correct += (int_preds == labels).sum().item()
		total += labels.size(0)

	return 100.0 * correct / max(total, 1)


def benchmark() -> dict:
	"""Run float and integer evaluation for all datasets."""
	_disable_inference_debug_trace()

	results = {}
	for name in ["MNIST", "CIFAR10", "Brain-MRI"]:
		loader = _get_test_loader(name)
		model = _build_model(name)
		float_acc = _float_accuracy(model, loader)
		int_acc = _integer_accuracy(model, loader)
		results[name] = {"float": float_acc, "integer": int_acc}
	return results


if __name__ == "__main__":
	metrics = benchmark()
	with open("benchmark_results.json", "w") as f:
		json.dump(metrics, f, indent=2)
	print("Saved benchmark_results.json with:")
	for ds, vals in metrics.items():
		print(f"  {ds}: float={vals['float']:.2f}%, integer={vals['integer']:.2f}%")
