"""Benchmark float vs integer inference on MNIST, CIFAR10, and Brain-MRI.

Runs the same models and integer pipeline as inference.py over the
corresponding 10% test partitions defined in resnet18.py and saves
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
import resnet18 as train_mod


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


def _integer_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
	"""Compute accuracy using the integer ResNet18 pipeline from inference.py."""
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

		# 3) Integer forward pass (mirrors resnet/inference.main)
		# Initial conv1
		q_x, s_out, z_out = inference.run_integer_conv_block(
			q_x,
			model.conv1,
			model.bn1,
			"conv1",
			scale_in,
			zp_in,
			apply_relu=True,
		)

		# Traverse all residual blocks
		for layer_idx, stage in enumerate(
			[model.layer1, model.layer2, model.layer3, model.layer4], 1
		):
			for block_idx, block in enumerate(stage):
				prefix = f"layer{layer_idx}_block{block_idx}"
				q_x, s_out, z_out = inference.run_integer_basic_block(
					q_x, block, prefix, s_out, z_out
				)

		# Global Average Pooling in float domain
		feat_float = s_out * (q_x.to(torch.float32) - z_out)
		feat_float = torch.nn.functional.adaptive_avg_pool2d(feat_float, (1, 1))
		feat_float = feat_float.view(feat_float.size(0), -1)

		# Quantize pooled features before FC
		pseudo_fc_in = torch.tensor(
			[feat_float.min().item(), feat_float.max().item()], dtype=torch.float32
		)
		s_fc_in, z_fc_in = inference.get_quantization_params(
			pseudo_fc_in, num_bits=8
		)
		q_fc_in = inference.quantize_tensor(
			feat_float, s_fc_in, z_fc_in, dtype=torch.uint8
		)

		q_out, final_s, final_z, _, _, _ = inference.run_integer_fc(
			q_fc_in, model.fc, "fc", s_fc_in, z_fc_in
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

