"""Offline calibration for UNet INT8 export.

This follows the existing calibration pattern used in the other model folders,
but it pulls the evaluation split directly from ``u_net.setup_data`` so the
calibration pass always uses the same deterministic 80/10/10 split as training.
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Optional

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INT8_DIR = os.path.join(THIS_DIR, "INT8")

if THIS_DIR not in sys.path:
	sys.path.insert(0, THIS_DIR)

import u_net as train_mod


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


int8_inference = _load_module(
	"unet_int8_inference",
	os.path.join(INT8_DIR, "inference.py"),
	INT8_DIR,
)
int8_utils = _load_module(
	"unet_int8_utils",
	os.path.join(INT8_DIR, "utils.py"),
	INT8_DIR,
)


SUPPORTED_DATASETS = (
	"Skin-Lesion",
	"Flood",
	"Brain-MRI-Seg",
	"BUSI",
)


def _normalize_dataset_name(dataset_name: str) -> str:
	key = dataset_name.strip().upper().replace("_", "-").replace(" ", "-")
	if key in {"SKIN-LESION", "SKINLESION"}:
		return "Skin-Lesion"
	if key == "FLOOD":
		return "Flood"
	if key in {"BRAIN-MRI-SEG", "BRAIN_MRI", "BRAINMRISEG"}:
		return "Brain-MRI-Seg"
	if key == "BUSI":
		return "BUSI"
	raise ValueError(f"Unknown dataset: {dataset_name}")


def _dataset_config(dataset_name: str, activation: str):
	display = _normalize_dataset_name(dataset_name)
	model_path_map = {
		"Skin-Lesion": f"best_unet5_{activation}_skin_lesion.pth",
		"Flood": f"best_unet5_{activation}_flood.pth",
		"Brain-MRI-Seg": f"best_unet5_{activation}_brain_mri_seg.pth",
		"BUSI": f"best_unet5_{activation}_busi.pth",
	}

	if display not in model_path_map:
		raise ValueError(f"Unsupported dataset: {dataset_name}")

	return {
		"display": display,
		"setup_fn": train_mod.setup_data,
		"model": train_mod.UNet(n_class=1, activation=activation),
		"model_path": os.path.join(THIS_DIR, model_path_map[display]),
		"download_name": display,
	}


def _sample_loader_dataset(loader: torch.utils.data.DataLoader, max_samples: int = 1000, seed: int = 42):
	dataset = loader.dataset
	sample_count = min(max_samples, len(dataset))
	if sample_count == len(dataset):
		return loader

	generator = torch.Generator().manual_seed(seed)
	indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
	sampled_dataset = torch.utils.data.Subset(dataset, indices)
	return torch.utils.data.DataLoader(
		sampled_dataset,
		batch_size=loader.batch_size,
		shuffle=False,
		num_workers=getattr(loader, "num_workers", 0),
		pin_memory=getattr(loader, "pin_memory", False),
	)

def _get_train_loader(cfg, batch_size: int, image_size: int):
	train_mod.train_loader = None
	train_mod.val_loader = None
	train_mod.test_loader = None

	setup_result = cfg["setup_fn"](
		train_data=cfg["download_name"],
		batch_size=batch_size,
		image_size=image_size,
		num_workers=0,
	)

	if (
		isinstance(setup_result, tuple)
		and len(setup_result) >= 3
		and hasattr(setup_result[0], "dataset")
	):
		loader = _sample_loader_dataset(setup_result[0], max_samples=1000)
		return loader

	if train_mod.train_loader is not None:
		sampled_loader = _sample_loader_dataset(train_mod.train_loader, max_samples=1000)
		return sampled_loader

	raise RuntimeError(f"Could not resolve train loader for dataset: {cfg['display']}")


def _output_file_name(dataset_name: str, activation: str) -> str:
	return f"{dataset_name.lower().replace(' ', '_').replace('-', '_')}_{activation}_calibration.json"


# -----------------------------------
# Local Calibration Hooks
# -----------------------------------
activation_ranges = {}

def register_hooks(model):
    handles = []
    activation_ranges.clear()

    def get_hook(name):
        def hook(model, input, output):
            in_tensor = input[0].detach()
            out_tensor = output.detach()
            
            if name not in activation_ranges:
                activation_ranges[name] = {
                    "in_min": in_tensor.min().item(),
                    "in_max": in_tensor.max().item(),
                    "out_min": out_tensor.min().item(),
                    "out_max": out_tensor.max().item(),
                }
            else:
                activation_ranges[name]["in_min"] = min(activation_ranges[name]["in_min"], in_tensor.min().item())
                activation_ranges[name]["in_max"] = max(activation_ranges[name]["in_max"], in_tensor.max().item())
                activation_ranges[name]["out_min"] = min(activation_ranges[name]["out_min"], out_tensor.min().item())
                activation_ranges[name]["out_max"] = max(activation_ranges[name]["out_max"], out_tensor.max().item())
        return hook

    # Dynamically attach hooks to UNet's Convolutions and Transposed Convolutions
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(get_hook(name)))
            
    return handles

def _register_and_collect(model: torch.nn.Module):
    return register_hooks(model)


def main(dataset_name: str, batch_size: int = 1, image_size: int = 256, out_dir: Optional[str] = None, activation: str = "relu"):
	cfg = _dataset_config(dataset_name, activation)
	display = cfg["display"]
	
	if not os.path.exists(cfg["model_path"]):
		raise FileNotFoundError(f"Model weights not found for {display} with activation '{activation}'. Please train the model first. Expected: {cfg['model_path']}")

	print(f"[calib] Dataset: {display} | Activation: {activation} - preparing train split (batch_size={batch_size}, image_size={image_size})")
	train_mod.dataset_downloader(cfg["download_name"])
	loader = _get_train_loader(cfg, batch_size=batch_size, image_size=image_size)

	model = cfg["model"]
	state = torch.load(cfg["model_path"], map_location="cpu")
	if len(state) > 0 and list(state.keys())[0].startswith("module."):
		state = {key[7:]: value for key, value in state.items()}
	model.load_state_dict(state)
	model.eval()

	aggregated: Dict[str, Dict[str, float]] = {}
	total = len(loader.dataset)
	print(f"[calib] Running forward passes over {total} test samples...")

	with torch.no_grad():
		for idx, (images, _masks) in enumerate(loader, 1):
			handles = _register_and_collect(model)
			try:
				_ = model(images)
			finally:
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
				else:
					current = aggregated[name]
					current["in_min"] = min(current["in_min"], float(ranges["in_min"]))
					current["in_max"] = max(current["in_max"], float(ranges["in_max"]))
					current["out_min"] = min(current["out_min"], float(ranges["out_min"]))
					current["out_max"] = max(current["out_max"], float(ranges["out_max"]))

			if idx % 25 == 0 or idx == total:
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
	out_path = os.path.join(out_dir, _output_file_name(display, activation))
	with open(out_path, "w") as f:
		json.dump(
			{
				"dataset": display,
				"split": "train_sampled",
				"activation": activation,
				"layers": calib,
			},
			f,
			indent=2,
		)

	print(f"[calib] Saved calibration to: {out_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--train_data",
		type=str,
		default=None,
		help="Training dataset to calibrate (Skin-Lesion, Flood, Brain-MRI-Seg, BUSI). If omitted, calibrate all supported datasets.",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="Batch size to use while running forward passes (default: 1)",
	)
	parser.add_argument(
		"--image_size",
		type=int,
		default=256,
		help="Input image size used by u_net.setup_data (default: 256)",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default=None,
		help="Optional output directory for the calibration JSON",
	)

	parser.add_argument(
		"--activation",
		type=str,
		default=None,
		choices=["relu", "gelu", "leaky_relu"],
		help="Activation function the model was trained with"
	)

	args = parser.parse_args()

	targets = [args.train_data] if args.train_data else SUPPORTED_DATASETS
	activations = [args.activation] if args.activation else ["relu", "gelu", "leaky_relu"]
	
	for dataset_name in targets:
		for activation in activations:
			print(f"\n[calib] === Calibrating dataset: {dataset_name} | Activation: {activation} ===")
			main(dataset_name, batch_size=args.batch_size, image_size=args.image_size, out_dir=args.out_dir, activation=activation)
