"""Benchmark float vs fixed-point inference on MNIST, CIFAR10, and Brain-MRI.

Runs the same models and Static 64-bit Fixed-Point pipeline as inference.py
over the corresponding 10% test partitions defined in resnet18.py and saves
accuracies to benchmark_results.json in the form:

{
	"MNIST": {"float": acc_float, "fixed_point": acc_int},
	"CIFAR10": {"float": acc_float, "fixed_point": acc_int},
	"Brain-MRI": {"float": acc_float, "fixed_point": acc_int}
}
"""

import json

import torch
from torch.utils.data import DataLoader

import inference
import resnet18 as train_mod
from utils import quantize_fixed_point, dequantize_fixed_point


def _disable_inference_debug_trace():
	"""Turn off heavy debug logging in inference when benchmarking."""

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
	"""Return the test DataLoader using resnet18.py's 80/10/10 split.

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
	"""Construct the trained ResNet18 model corresponding to the dataset."""
	name = dataset_name.upper()
	if name == "MNIST":
		model = inference.ResNet18Inference(num_classes=10, in_channels=1)
		model_path = "best_resnet18_mnist.pth"
	elif name == "BRAIN-MRI":
		# Note: adjust in_channels if Brain-MRI was trained as RGB
		model = inference.ResNet18Inference(num_classes=4, in_channels=1)
		model_path = "best_resnet18_brain_mri.pth"
	elif name in ("CIFR10", "CIFAR10"):
		model = inference.ResNet18Inference(num_classes=10, in_channels=3)
		model_path = "best_resnet18_cifar10.pth"
	else:
		raise ValueError(f"Unknown dataset: {dataset_name}")

	state = torch.load(model_path, map_location="cpu")
	# Handle possible DataParallel checkpoints
	if list(state.keys())[0].startswith("module."):
		state = {k[7:]: v for k, v in state.items()}

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


def _fixed_point_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
	"""Compute accuracy using the static 64-bit fixed-point ResNet18 pipeline."""
	correct = 0
	total = 0

	for images, labels in loader:
		# 1) Quantize input directly to static Q31.32 int64
		q_x = quantize_fixed_point(images)

		# 2) Static fixed-point forward pass (mirrors resnet/inference.main)
		q_x = inference.run_static_fixed_point_conv_block(
			q_x, model.conv1, model.bn1, apply_relu=True
		)

		for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
			for block in stage:
				q_x = inference.run_static_fixed_point_basic_block(q_x, block)

		q_pooled = inference.fixed_point_global_avg_pool2d(q_x)
		q_fc_in = q_pooled.view(q_pooled.size(0), -1)

		q_out, _, _ = inference.run_static_fixed_point_fc(q_fc_in, model.fc)

		# 3) Dequantize logits and compute predictions
		dequantized_logits = dequantize_fixed_point(q_out)
		int_preds = dequantized_logits.argmax(dim=1)

		correct += (int_preds == labels).sum().item()
		total += labels.size(0)

	return 100.0 * correct / max(total, 1)


def benchmark() -> dict:
	"""Run float and fixed point evaluation for all datasets."""
	_disable_inference_debug_trace()

	results = {}
	for name in ["MNIST", "CIFAR10", "Brain-MRI"]:
		loader = _get_test_loader(name)
		model = _build_model(name)
		float_acc = _float_accuracy(model, loader)
		fixed_point_acc = _fixed_point_accuracy(model, loader)
		results[name] = {"float": float_acc, "fixed_point": fixed_point_acc}
	return results


if __name__ == "__main__":
	metrics = benchmark()
	with open("benchmark_results.json", "w") as f:
		json.dump(metrics, f, indent=2)
	print("Saved benchmark_results.json with:")
	for ds, vals in metrics.items():
		print(f"  {ds}: float={vals['float']:.2f}%, fixed_point={vals['fixed_point']:.2f}%")

