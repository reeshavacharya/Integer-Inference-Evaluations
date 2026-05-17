"""Layerwise quantization error analysis for LeNet INT8 inference."""

import argparse
import importlib.util
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
INT8_DIR = os.path.join(THIS_DIR, "INT8")
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import lenet5 as train_mod


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
    "lenet_int8_inference", os.path.join(INT8_DIR, "inference.py"), INT8_DIR
)
int8_utils = _load_module(
    "lenet_int8_utils", os.path.join(INT8_DIR, "utils.py"), INT8_DIR
)


class LayerMetrics:
    def __init__(self):
        self.cos_sim_sum = 0.0
        self.sqnr_sum = 0.0
        self.mae_sum = 0.0
        self.mean_shift_sum = 0.0
        self.max_abs_err = 0.0
        self.batches = 0

    def update(self, fp_tensor, dq_tensor):
        fp_flat = fp_tensor.detach().view(-1).float()
        dq_flat = dq_tensor.detach().view(-1).float()

        # Force 64-bit precision to prevent Cauchy-Schwarz violation on dense medical data
        fp_64 = fp_flat.to(torch.float64)
        dq_64 = dq_flat.to(torch.float64)
        cos_sim = F.cosine_similarity(fp_64, dq_64, dim=0).item()
        
        signal_power = torch.sum(fp_flat**2)
        noise_power = torch.sum((fp_flat - dq_flat) ** 2)
        sqnr = (
            (10 * torch.log10(signal_power / noise_power)).item()
            if noise_power > 1e-7
            else 100.0
        )

        abs_diff = torch.abs(fp_flat - dq_flat)
        mae = abs_diff.mean().item()
        batch_max_err = abs_diff.max().item()
        mean_shift = (fp_flat.mean() - dq_flat.mean()).item()

        self.cos_sim_sum += cos_sim
        self.sqnr_sum += sqnr
        self.mae_sum += mae
        self.mean_shift_sum += mean_shift
        self.max_abs_err = max(self.max_abs_err, batch_max_err)
        self.batches += 1

    def finalize(self):
        if self.batches == 0:
            return {}
        return {
            "cosine_similarity": self.cos_sim_sum / self.batches,
            "sqnr_db": self.sqnr_sum / self.batches,
            "mean_absolute_error": self.mae_sum / self.batches,
            "max_absolute_error": self.max_abs_err,
            "mean_shift": self.mean_shift_sum / self.batches,
        }


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


def _calibration_file_name(dataset_display: str):
    return f"{dataset_display.lower().replace(' ', '_').replace('-', '_')}_calibration.json"


def _calibration_file_path(dataset_display: str):
    return os.path.join(
        THIS_DIR, "calibration", _calibration_file_name(dataset_display)
    )


def _resolve_test_loader(setup_fn, batch_size: int):
    train_mod.train_loader = None
    train_mod.val_loader = None
    train_mod.test_loader = None

    setup_result = setup_fn(batch_size=batch_size)

    if (
        isinstance(setup_result, tuple)
        and len(setup_result) >= 3
        and hasattr(setup_result[2], "dataset")
    ):
        return setup_result[2]

    if train_mod.test_loader is not None:
        return train_mod.test_loader

    raise RuntimeError("Could not resolve test loader")


def _run_fp32_trace(model, images):
    traces = {}
    x = images

    if isinstance(model, train_mod.MedicalLeNet):
        x = model.features[0](x)
        x = model.features[1](x)
        x = model.features[2](x)
        x = model.features[3](x)
        traces["conv1"] = x.detach().clone()

        x = model.features[4](x)
        x = model.features[5](x)
        x = model.features[6](x)
        x = model.features[7](x)
        traces["conv2"] = x.detach().clone()

        x = model.classifier[0](x)
        x = model.classifier[1](x)
        x = model.classifier[2](x)
        traces["fc1"] = x.detach().clone()

        x = model.classifier[3](x)
        x = model.classifier[4](x)
        x = model.classifier[5](x)
        traces["fc2"] = x.detach().clone()

        x = model.classifier[6](x)
        x = model.classifier[7](x)
        traces["fc3"] = x.detach().clone()
        return traces

    x = model.features[0](x)
    x = model.features[1](x)
    x = model.features[2](x)
    traces["conv1"] = x.detach().clone()

    x = model.features[3](x)
    x = model.features[4](x)
    x = model.features[5](x)
    traces["conv2"] = x.detach().clone()

    x = model.classifier[0](x)
    x = model.classifier[1](x)
    x = model.classifier[2](x)
    traces["fc1"] = x.detach().clone()

    x = model.classifier[3](x)
    x = model.classifier[4](x)
    traces["fc2"] = x.detach().clone()

    x = model.classifier[5](x)
    traces["fc3"] = x.detach().clone()
    return traces


def _run_int8_trace(model, images, int8_state):
    traces = {}

    # 1. Pull global input boundaries from the compiled offline dictionary
    scale_in = int8_state["meta"]["in_scale"]
    zp_in = int8_state["meta"]["in_zp"]

    # 2. Quantize Input Image
    q_x = int8_utils.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

    # ---------------------------------------------------------
    # Execute Convs using int8_state
    # ---------------------------------------------------------
    q_x, s_out, z_out = int8_inference.run_integer_layer(
        q_x, int8_state["conv1"], "conv1", zp_in, apply_relu=True, is_conv=True
    )
    q_x = int8_inference.avg_pool_uint8(q_x, name="bench_pool_after_conv1")
    traces["conv1"] = s_out * (q_x.to(torch.float32) - z_out)

    q_x, s_out, z_out = int8_inference.run_integer_layer(
        q_x, int8_state["conv2"], "conv2", z_out, apply_relu=True, is_conv=True
    )
    q_x = int8_inference.avg_pool_uint8(q_x, name="bench_pool_after_conv2")
    traces["conv2"] = s_out * (q_x.to(torch.float32) - z_out)

    # Flatten for Dense Layers
    q_x = q_x.view(q_x.size(0), -1)

    # ---------------------------------------------------------
    # Execute FCs using int8_state
    # ---------------------------------------------------------
    q_x, s_out, z_out = int8_inference.run_integer_layer(
        q_x, int8_state["fc1"], "fc1", z_out, apply_relu=True, is_conv=False
    )
    traces["fc1"] = s_out * (q_x.to(torch.float32) - z_out)

    q_x, s_out, z_out = int8_inference.run_integer_layer(
        q_x, int8_state["fc2"], "fc2", z_out, apply_relu=True, is_conv=False
    )
    traces["fc2"] = s_out * (q_x.to(torch.float32) - z_out)

    q_out, final_s, final_z = int8_inference.run_integer_layer(
        q_x, int8_state["fc3"], "fc3", z_out, apply_relu=False, is_conv=False
    )
    traces["fc3"] = final_s * (q_out.to(torch.float32) - final_z)

    return traces


def evaluate_error(dataset_name: str, num_data: int = None, batch_size: int = 64):
    dataset_display = _normalize_dataset_name(dataset_name)
    cfg = int8_inference._resolve_infer_config(dataset_display)
    model = cfg["model"]

    state = torch.load(
        os.path.join(THIS_DIR, os.path.basename(cfg["model_path"])), map_location="cpu"
    )
    if len(state) > 0 and list(state.keys())[0].startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    int8_path = os.path.join(
        THIS_DIR, os.path.basename(cfg["model_path"]).replace(".pth", "_int8.pth")
    )
    if not os.path.exists(int8_path):
        raise FileNotFoundError(
            f"Missing compiled model: {int8_path}. Run INT8/export_int8_model.py first."
        )
    int8_state = torch.load(int8_path, map_location="cpu")

    train_mod.datasetDownloader(dataset_display)
    loader = _resolve_test_loader(cfg["setup_fn"], batch_size=batch_size)
    int8_inference.load_calibration_ranges(dataset_display)

    total_images = len(loader.dataset)
    target_images = total_images if num_data is None else min(num_data, total_images)
    processed_images = 0

    layer_trackers = {}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, 1):
            if processed_images >= target_images:
                break

            remaining = target_images - processed_images
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            fp32_traces = _run_fp32_trace(model, images)
            int8_traces = _run_int8_trace(model, images, int8_state)

            for name in ["conv1", "conv2", "fc1", "fc2", "fc3"]:
                if name not in layer_trackers:
                    layer_trackers[name] = LayerMetrics()
                layer_trackers[name].update(fp32_traces[name], int8_traces[name])

            processed_images += images.size(0)
            print(f"[error] Processed {processed_images}/{target_images} images...")

    return {name: tracker.finalize() for name, tracker in layer_trackers.items()}


def plot_error_metrics(metrics_dict, dataset_name):
    layers = list(metrics_dict.keys())
    valid_layers = [layer for layer in layers if metrics_dict[layer]]
    if not valid_layers:
        print("[-] No valid metrics to plot.")
        return

    mae = [metrics_dict[layer]["mean_absolute_error"] for layer in valid_layers]
    max_err = [metrics_dict[layer]["max_absolute_error"] for layer in valid_layers]
    sqnr = [metrics_dict[layer]["sqnr_db"] for layer in valid_layers]
    cos_sim = [metrics_dict[layer]["cosine_similarity"] for layer in valid_layers]
    mean_shift = [metrics_dict[layer]["mean_shift"] for layer in valid_layers]

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"LeNet5 FP32 vs INT8 Quantization Error Progression ({dataset_name})",
        fontsize=16,
        fontweight="bold",
    )

    x = range(len(valid_layers))
    axs[0, 0].plot(x, mae, marker="o", color="blue", label="Mean Absolute Error")
    axs[0, 0].plot(
        x, max_err, marker="x", color="red", linestyle="--", label="Max Absolute Error"
    )
    axs[0, 0].set_title("Error Magnitude Growth")
    axs[0, 0].set_ylabel("Absolute Error")
    axs[0, 0].grid(True, linestyle=":", alpha=0.7)
    axs[0, 0].legend()

    axs[0, 1].plot(x, sqnr, marker="s", color="purple")
    axs[0, 1].set_title("Signal-to-Quantization-Noise Ratio (SQNR)")
    axs[0, 1].set_ylabel("dB")
    axs[0, 1].axhline(y=20, color="r", linestyle="-", alpha=0.3)
    axs[0, 1].grid(True, linestyle=":", alpha=0.7)

    axs[1, 0].plot(x, cos_sim, marker="^", color="green")
    axs[1, 0].set_title("Cosine Similarity")
    axs[1, 0].set_ylabel("Similarity")
    axs[1, 0].set_ylim(min(0.85, min(cos_sim) - 0.05), 1.01)
    axs[1, 0].grid(True, linestyle=":", alpha=0.7)

    axs[1, 1].bar(
        x,
        mean_shift,
        color=["red" if value < 0 else "blue" for value in mean_shift],
        alpha=0.6,
    )
    axs[1, 1].axhline(y=0, color="black", linestyle="-")
    axs[1, 1].set_title("Mean Shift")
    axs[1, 1].set_ylabel("Shift Direction")
    axs[1, 1].grid(True, linestyle=":", alpha=0.7, axis="y")

    for ax in axs.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(valid_layers, rotation=45, ha="right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    filename = os.path.join(THIS_DIR, f"quantization_divergence_{safe_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[+] Saved error divergence graphs to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to evaluate")
    parser.add_argument(
        "--MNIST", dest="mnist", action="store_true", help="Evaluate MNIST"
    )
    parser.add_argument(
        "--CIFAR10", dest="cifar10", action="store_true", help="Evaluate CIFAR10"
    )
    parser.add_argument(
        "--Brain-MRI", dest="brain_mri", action="store_true", help="Evaluate Brain-MRI"
    )
    parser.add_argument(
        "--NIH-CHEST", dest="nih_chest", action="store_true", help="Evaluate NIH-CHEST"
    )
    parser.add_argument(
        "--OCTMNIST", dest="octmnist", action="store_true", help="Evaluate OCTMNIST"
    )
    parser.add_argument(
        "--BloodMNIST",
        dest="bloodmnist",
        action="store_true",
        help="Evaluate BloodMNIST",
    )
    parser.add_argument(
        "--OrganAMNIST",
        dest="organamnist",
        action="store_true",
        help="Evaluate OrganAMNIST",
    )
    parser.add_argument(
        "--num_data", type=int, default=None, help="Number of images to process"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for evaluation"
    )
    args = parser.parse_args()

    selected = []
    for enabled, dataset_name in [
        (args.mnist, "MNIST"),
        (args.cifar10, "CIFAR10"),
        (args.brain_mri, "Brain-MRI"),
        (args.nih_chest, "NIH-CHEST"),
        (args.octmnist, "OCTMNIST"),
        (args.bloodmnist, "BloodMNIST"),
        (args.organamnist, "OrganAMNIST"),
    ]:
        if enabled:
            selected.append(dataset_name)

    if args.dataset is not None:
        selected = [_normalize_dataset_name(args.dataset)]
    elif not selected:
        selected = [
            "MNIST",
            "CIFAR10",
            "Brain-MRI",
            "NIH-CHEST",
            "OCTMNIST",
            "BloodMNIST",
            "OrganAMNIST",
        ]

    for dataset_name in selected:
        print(f"\n[error] === Evaluating dataset: {dataset_name} ===")
        metrics = evaluate_error(
            dataset_name, args.num_data, batch_size=args.batch_size
        )

        safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        file_name = os.path.join(THIS_DIR, f"error_accumulation_{safe_name}.json")
        with open(file_name, "w") as f:
            json.dump({"dataset": dataset_name, "layer_metrics": metrics}, f, indent=2)

        print(f"\n[+] Saved error accumulation log to {file_name}")
        plot_error_metrics(metrics, dataset_name)
