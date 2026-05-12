"""Generate dataset-wide calibration ranges for VGG19 INT8 inference.

The script mirrors the ResNet calibration entrypoint:
- select a dataset explicitly with a flag or calibrate all supported datasets
- run deterministic test-split forward passes
- collect per-layer input/output min/max values
- save a dataset-named JSON file under VGGNet/calibration/
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INT8_DIR = os.path.join(THIS_DIR, "INT8")

if THIS_DIR not in sys.path:
	sys.path.insert(0, THIS_DIR)

import vgg19 as train_mod


SUPPORTED_DATASETS = (
	"MNIST",
	"CIFAR10",
	"Brain-MRI",
	"NIH-CHEST",
	"OCTMNIST",
	"BloodMNIST",
	"OrganAMNIST",
)


def _load_module(module_name: str, file_path: str, prepend_dir: str):
	saved_path = list(sys.path)
	previous_utils = sys.modules.get("utils")
	try:
		sys.path.insert(0, prepend_dir)
		spec = importlib.util.spec_from_file_location(module_name, file_path)
		if spec is None or spec.loader is None:
			raise RuntimeError(f"Failed to create module spec for {file_path}")
		module = importlib.util.module_from_spec(spec)
		sys.modules[module_name] = module
		spec.loader.exec_module(module)
		return module
	finally:
		sys.path = saved_path
		if previous_utils is None:
			sys.modules.pop("utils", None)
		else:
			sys.modules["utils"] = previous_utils


int8_utils = _load_module("vgg_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR)


def _normalize_dataset_name(dataset_name: str) -> str:
	key = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
	if key == "CIFR10":
		return "CIFAR10"
	if key == "BRAIN-MRI":
		return "Brain-MRI"
	if key == "NIH-CHEST":
		return "NIH-CHEST"
	if key == "MNIST":
		return "MNIST"
	if key == "CIFAR10":
		return "CIFAR10"
	if key == "OCTMNIST":
		return "OCTMNIST"
	if key == "BLOODMNIST":
		return "BloodMNIST"
	if key == "ORGANAMNIST":
		return "OrganAMNIST"
	raise ValueError(f"Unknown dataset: {dataset_name}")


def _dataset_config(dataset_name: str):
	display = _normalize_dataset_name(dataset_name)
	if display == "MNIST":
		return {
			"display": display,
			"setup_fn": train_mod.setup_MNIST,
			"model": train_mod.VGG19(num_classes=10, in_channels=1),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_mnist.pth"),
			"download_name": "MNIST",
		}
	if display == "CIFAR10":
		return {
			"display": display,
			"setup_fn": train_mod.setup_CIFAR10,
			"model": train_mod.VGG19(num_classes=10, in_channels=3),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_cifar10.pth"),
			"download_name": "CIFAR10",
		}
	if display == "Brain-MRI":
		return {
			"display": display,
			"setup_fn": train_mod.setup_Brain_MRI,
			"model": train_mod.VGG19(num_classes=4, in_channels=1),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_brain_mri.pth"),
			"download_name": "Brain-MRI",
		}
	if display == "NIH-CHEST":
		return {
			"display": display,
			"setup_fn": train_mod.setup_NIH_Chest,
			"model": train_mod.VGG19(num_classes=15, in_channels=1),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_NIH_Chest_XRay.pth"),
			"download_name": "NIH-CHEST",
		}
	if display == "OCTMNIST":
		return {
			"display": display,
			"setup_fn": train_mod.setup_OCTMNIST,
			"model": train_mod.VGG19(num_classes=4, in_channels=1),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_octmnist.pth"),
			"download_name": "OCTMNIST",
		}
	if display == "BloodMNIST":
		return {
			"display": display,
			"setup_fn": train_mod.setup_BloodMNIST,
			"model": train_mod.VGG19(num_classes=8, in_channels=3),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_bloodmnist.pth"),
			"download_name": "BloodMNIST",
		}
	if display == "OrganAMNIST":
		return {
			"display": display,
			"setup_fn": train_mod.setup_OrganAMNIST,
			"model": train_mod.VGG19(num_classes=11, in_channels=1),
			"model_path": os.path.join(THIS_DIR, "best_vgg19_organamnist.pth"),
			"download_name": "OrganAMNIST",
		}
	raise ValueError(f"Unsupported dataset: {dataset_name}")


def _resolve_test_loader(cfg, batch_size: int):
	train_mod.train_loader = None
	train_mod.val_loader = None
	train_mod.test_loader = None

	setup_result = cfg["setup_fn"](batch_size=batch_size)

	if (
		isinstance(setup_result, tuple)
		and len(setup_result) >= 3
		and hasattr(setup_result[2], "dataset")
	):
		return setup_result[2]

	if train_mod.test_loader is not None:
		return train_mod.test_loader

	raise RuntimeError(f"Could not resolve test loader for dataset: {cfg['display']}")


activation_ranges: Dict[str, Dict[str, float]] = {}


def _register_hooks(model: nn.Module):
    handles = []

    def hook_in(name):
        def hook(module, inputs, output):
            in_min, in_max = float(inputs[0].min()), float(inputs[0].max())
            if name not in activation_ranges:
                activation_ranges[name] = {
                    "in_min": in_min, "in_max": in_max, 
                    "out_min": float('inf'), "out_max": float('-inf')
                }
            else:
                activation_ranges[name]["in_min"] = min(activation_ranges[name]["in_min"], in_min)
                activation_ranges[name]["in_max"] = max(activation_ranges[name]["in_max"], in_max)
        return hook

    def hook_out(name):
        def hook(module, inputs, output):
            out_min, out_max = float(output.min()), float(output.max())
            if name not in activation_ranges:
                activation_ranges[name] = {
                    "in_min": float('inf'), "in_max": float('-inf'), 
                    "out_min": out_min, "out_max": out_max
                }
            else:
                activation_ranges[name]["out_min"] = min(activation_ranges[name]["out_min"], out_min)
                activation_ranges[name]["out_max"] = max(activation_ranges[name]["out_max"], out_max)
        return hook

    # 1. Hook Features (Grab input at Conv, grab output at following ReLU)
    last_conv_name = None
    for idx, module in enumerate(model.features):
        if isinstance(module, nn.Conv2d):
            last_conv_name = f"features_{idx}"
            handles.append(module.register_forward_hook(hook_in(last_conv_name)))
        elif isinstance(module, nn.ReLU) and last_conv_name:
            handles.append(module.register_forward_hook(hook_out(last_conv_name)))
            last_conv_name = None

    # 2. Hook Adaptive Avg Pool
    def pool_hook(name):
        def hook(module, inputs, output):
            in_t, out_t = inputs[0].detach(), output.detach()
            if name not in activation_ranges:
                activation_ranges[name] = {
                    "in_min": float(in_t.min()), "in_max": float(in_t.max()),
                    "out_min": float(out_t.min()), "out_max": float(out_t.max())
                }
            else:
                activation_ranges[name]["in_min"] = min(activation_ranges[name]["in_min"], float(in_t.min()))
                activation_ranges[name]["in_max"] = max(activation_ranges[name]["in_max"], float(in_t.max()))
                activation_ranges[name]["out_min"] = min(activation_ranges[name]["out_min"], float(out_t.min()))
                activation_ranges[name]["out_max"] = max(activation_ranges[name]["out_max"], float(out_t.max()))
        return hook
    handles.append(model.avgpool.register_forward_hook(pool_hook("avgpool")))

    # 3. Hook Classifier (Grab input at Linear, grab output at following ReLU/Dropout)
    last_fc_name = None
    for idx, module in enumerate(model.classifier):
        if isinstance(module, nn.Linear):
            last_fc_name = f"classifier_{idx}"
            handles.append(module.register_forward_hook(hook_in(last_fc_name)))
            
            # The final Linear layer has no ReLU after it, so we hook its own output
            if idx == len(model.classifier) - 1:
                handles.append(module.register_forward_hook(hook_out(last_fc_name)))
                
        elif isinstance(module, nn.ReLU) and last_fc_name:
            handles.append(module.register_forward_hook(hook_out(last_fc_name)))
            last_fc_name = None

    return handles


def _load_model(cfg):
	model = cfg["model"]
	state = torch.load(cfg["model_path"], map_location="cpu")
	if list(state.keys())[0].startswith("module."):
		state = {key[7:]: value for key, value in state.items()}
	model.load_state_dict(state)
	model.eval()
	return model


def _output_file_name(dataset_name: str) -> str:
	return f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_calibration.json"


def main(dataset_name: str, batch_size: int = 1, out_dir: Optional[str] = None):
	cfg = _dataset_config(dataset_name)
	display = cfg["display"]

	print(f"[calib] Dataset: {display} - preparing calibration loader (batch_size={batch_size})")
	train_mod.datasetDownloader(cfg["download_name"])
	loader = _resolve_test_loader(cfg, batch_size=batch_size)

	model = _load_model(cfg)
	aggregated: Dict[str, Dict[str, float]] = {}

	total = len(loader.dataset)
	print(f"[calib] Running forward passes over {total} test samples...")

	with torch.no_grad():
		for idx, (images, _labels) in enumerate(loader, 1):
			activation_ranges.clear()
			handles = _register_hooks(model)
			_ = model(images)
			for handle in handles:
				handle.remove()

			for name, ranges in activation_ranges.items():
				if name not in aggregated:
					aggregated[name] = {
						"in_min": float(ranges["in_min"]),
						"in_max": float(ranges["in_max"]),
						"out_min": float(ranges["out_min"]),
						"out_max": float(ranges["out_max"]),
					}
					continue

				current = aggregated[name]
				current["in_min"] = min(current["in_min"], float(ranges["in_min"]))
				current["in_max"] = max(current["in_max"], float(ranges["in_max"]))
				current["out_min"] = min(current["out_min"], float(ranges["out_min"]))
				current["out_max"] = max(current["out_max"], float(ranges["out_max"]))

			if idx % 100 == 0 or idx == total:
				print(f"[calib] Processed {idx}/{total} samples")

	calib = {}
	for name, ranges in aggregated.items():
		out_tensor = torch.tensor([ranges["out_min"], ranges["out_max"]], dtype=torch.float32)
		in_tensor = torch.tensor([ranges["in_min"], ranges["in_max"]], dtype=torch.float32)
		out_scale, out_zp = int8_utils.get_quantization_params(out_tensor, num_bits=8)
		in_scale, in_zp = int8_utils.get_quantization_params(in_tensor, num_bits=8)
		calib[name] = {
			"in_min": ranges["in_min"],
			"in_max": ranges["in_max"],
			"in_scale": float(in_scale),
			"in_zero_point": int(in_zp),
			"out_min": ranges["out_min"],
			"out_max": ranges["out_max"],
			"out_scale": float(out_scale),
			"out_zero_point": int(out_zp),
		}

	out_dir = out_dir or os.path.join(THIS_DIR, "calibration")
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, _output_file_name(display))
	with open(out_path, "w") as f:
		json.dump({"dataset": display, "layers": calib}, f, indent=2)

	print(f"[calib] Saved calibration to: {out_path}")


def _selected_datasets_from_args(args: argparse.Namespace) -> Iterable[str]:
	selected = []
	flag_map = [
		(args.mnist, "MNIST"),
		(args.cifar10, "CIFAR10"),
		(args.brain_mri, "Brain-MRI"),
		(args.nih_chest, "NIH-CHEST"),
		(args.octmnist, "OCTMNIST"),
		(args.bloodmnist, "BloodMNIST"),
		(args.organamnist, "OrganAMNIST"),
	]

	for enabled, dataset_name in flag_map:
		if enabled:
			selected.append(dataset_name)

	if selected:
		return selected

	if args.dataset is not None:
		return [_normalize_dataset_name(args.dataset)]

	return SUPPORTED_DATASETS


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset",
		type=str,
		default=None,
		help="Dataset to calibrate (e.g. MNIST, CIFAR10, Brain-MRI, NIH-CHEST, OCTMNIST, BloodMNIST, OrganAMNIST). If omitted, calibrate all supported datasets.",
	)
	parser.add_argument(
		"--MNIST",
		dest="mnist",
		action="store_true",
		help="Calibrate the MNIST model",
	)
	parser.add_argument(
		"--CIFAR10",
		dest="cifar10",
		action="store_true",
		help="Calibrate the CIFAR10 model",
	)
	parser.add_argument(
		"--Brain-MRI",
		dest="brain_mri",
		action="store_true",
		help="Calibrate the Brain-MRI model",
	)
	parser.add_argument(
		"--NIH-CHEST",
		dest="nih_chest",
		action="store_true",
		help="Calibrate the NIH-CHEST model",
	)
	parser.add_argument(
		"--OCTMNIST",
		dest="octmnist",
		action="store_true",
		help="Calibrate the OCTMNIST model",
	)
	parser.add_argument(
		"--BloodMNIST",
		dest="bloodmnist",
		action="store_true",
		help="Calibrate the BloodMNIST model",
	)
	parser.add_argument(
		"--OrganAMNIST",
		dest="organamnist",
		action="store_true",
		help="Calibrate the OrganAMNIST model",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1,
		help="Batch size to use while running forward passes (default: 1)",
	)
	parser.add_argument(
		"--out-dir",
		type=str,
		default=None,
		help="Optional output directory for the calibration JSON",
	)

	args = parser.parse_args()

	for dataset_name in _selected_datasets_from_args(args):
		print(f"\n[calib] === Calibrating dataset: {dataset_name} ===")
		main(dataset_name, batch_size=args.batch_size, out_dir=args.out_dir)
