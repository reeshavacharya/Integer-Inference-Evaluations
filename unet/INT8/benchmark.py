"""Benchmark float vs integer inference on supported UNet datasets.

Runs the same models and integer pipeline as inference.py over the
corresponding 10% test splits (from u_net.setup_data) and saves
accuracies to benchmark_results.json in the form:

{
    "Skin-Lesion": {
        "float": {"dice": acc_float, "iou": acc_float, "acc": acc_float, "f1": acc_float},
        "integer": {"dice": acc_int, "iou": acc_int, "acc": acc_int, "f1": acc_int}
    },
    "Flood": {
        "float": {"dice": acc_float, "iou": acc_float, "acc": acc_float, "f1": acc_float},
        "integer": {"dice": acc_int, "iou": acc_int, "acc": acc_int, "f1": acc_int}
    },
}
"""

import json
from typing import Optional

import torch

import inference
import u_net


def _disable_inference_debug_trace() -> None:
    """Turn off heavy debug logging in inference when benchmarking.

    This prevents run_integer_layer/pool_uint8 from storing full tensors
    in inference.debug_trace while we iterate over the test set.
    """

    class _NoOpList(list):
        def append(self, item):  # type: ignore[override]
            # Discard logged items to keep memory usage low
            return None

    inference.debug_trace = {  # type: ignore[attr-defined]
        "input": None,
        "layers": _NoOpList(),
        "pooling": _NoOpList(),
    }


def _get_test_loader(dataset_name: str, batch_size: int = 4):
    """Return the test DataLoader for the given dataset (80/10/10 split).

    Uses u_net.setup_data, which performs an 80/10/10 train/val/test split
    with a fixed seed; we only need the test loader here.
    """

    _, _, test_loader = u_net.setup_data(
        train_data=dataset_name, batch_size=batch_size, image_size=256
    )
    return test_loader


def _build_model(dataset_name: str) -> torch.nn.Module:
    """Load the trained UNet model for the requested dataset."""

    model = inference.UNet(num_classes=1)
    if dataset_name == "Skin-Lesion":
        model_path = "best_unet5_skin_lesion.pth"
    elif dataset_name == "Flood":
        model_path = "best_unet5_flood.pth"
    elif dataset_name == "Brain-MRI-Seg":
        model_path = "best_unet5_brain_mri_seg.pth"
    elif dataset_name == "BUSI":
        model_path = "best_unet5_busi.pth"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_metrics_from_logits(logits: torch.Tensor, masks: torch.Tensor):
    """Compute Dice, IoU, accuracy, and F1 for binary segmentation logits.

    Expects logits and masks of shape [N, 1, H, W].
    """

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        tp = (preds_flat * masks_flat).sum().item()
        tn = ((1 - preds_flat) * (1 - masks_flat)).sum().item()
        fp = (preds_flat * (1 - masks_flat)).sum().item()
        fn = ((1 - preds_flat) * masks_flat).sum().item()

        eps = 1e-7
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    return dice, iou, acc, f1

def _float_metrics(
	model: torch.nn.Module,
	loader,
	dataset_name: str,
	num_data: Optional[int] = None,
) -> dict:
    """Compute average float Dice, IoU, accuracy, F1 over the test loader.

    If num_data is provided, only the first num_data samples from the loader
    are used for a quick benchmark.
    """
    total_samples = len(loader.dataset)
    target_samples = total_samples if num_data is None else min(num_data, total_samples)
    print(
        f"[bench][{dataset_name}][float] Starting benchmark over {target_samples}/{total_samples} samples."
    )

    # Accumulate confusion counts across all batches
    tp = tn = fp = fn = 0.0
    seen = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader, 1):
            if num_data is not None and seen >= num_data:
                break

            if num_data is not None and seen + images.size(0) > num_data:
                keep = num_data - seen
                images = images[:keep]
                masks = masks[:keep]

            logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            tp += (preds_flat * masks_flat).sum().item()
            tn += ((1 - preds_flat) * (1 - masks_flat)).sum().item()
            fp += (preds_flat * (1 - masks_flat)).sum().item()
            fn += ((1 - preds_flat) * masks_flat).sum().item()

            seen += images.size(0)
            remaining = max(target_samples - seen, 0)
            print(
                f"[bench][{dataset_name}][float] Batch {batch_idx}: processed {seen}/{target_samples} samples, remaining {remaining}."
            )

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    return {"dice": dice, "iou": iou, "acc": acc, "f1": f1}


def _integer_metrics(
    model: torch.nn.Module,
    loader,
    dataset_name: str,
    num_data: Optional[int] = None,
) -> dict:
    """Compute average integer Dice, IoU, accuracy, F1 using integer UNet path.

    This mirrors the integer-only path in inference.main, but runs over
    all batches from the 10% test DataLoader.
    """

    tp = tn = fp = fn = 0.0
    total_samples = len(loader.dataset)
    target_samples = total_samples if num_data is None else min(num_data, total_samples)
    print(
        f"[bench][{dataset_name}][integer] Starting benchmark over {target_samples}/{total_samples} samples."
    )

    seen = 0

    for batch_idx, (images, masks) in enumerate(loader, 1):
        if num_data is not None and seen >= num_data:
            break

        if num_data is not None and seen + images.size(0) > num_data:
            keep = num_data - seen
            images = images[:keep]
            masks = masks[:keep]

        # 1) Calibration for this batch (float forward pass with hooks)
        inference.activation_ranges.clear()
        handles = inference.register_hooks(model)
        with torch.no_grad():
            _ = model(images)
        for h in handles:
            h.remove()

        # 2) Quantize input using conv1 input range
        in_range = inference.activation_ranges["conv1"]
        pseudo_in_tensor = torch.tensor(
            [in_range["in_min"], in_range["in_max"]]
        )
        scale_in, zp_in = inference.get_quantization_params(
            pseudo_in_tensor, num_bits=8
        )
        q_x = inference.quantize_tensor(images, scale_in, zp_in, dtype=torch.uint8)

        # 3) Integer forward pass (mirrors inference.main integer path)
        cfg = inference._get_layer_config(model)

        # Encoder block 1
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv1"], "conv1", scale_in, zp_in, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv2"], "conv2", s, z, apply_relu=True
        )
        q_e12, s_e12, z_e12 = q_x, s, z
        q_x = inference.pool_uint8(q_x, name="pool1")

        # Encoder block 2
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv3"], "conv3", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv4"], "conv4", s, z, apply_relu=True
        )
        q_e22, s_e22, z_e22 = q_x, s, z
        q_x = inference.pool_uint8(q_x, name="pool2")

        # Encoder block 3
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv5"], "conv5", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv6"], "conv6", s, z, apply_relu=True
        )
        q_e32, s_e32, z_e32 = q_x, s, z
        q_x = inference.pool_uint8(q_x, name="pool3")

        # Encoder block 4
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv7"], "conv7", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv8"], "conv8", s, z, apply_relu=True
        )
        q_e42, s_e42, z_e42 = q_x, s, z
        q_x = inference.pool_uint8(q_x, name="pool4")

        # Bottom
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv9"], "conv9", s, z, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv10"], "conv10", s, z, apply_relu=True
        )

        # Decoder block 1 (skip from conv8)
        q_x, s_up1, z_up1, _, _, _ = inference.run_integer_layer(
            q_x, cfg["upconv1"], "upconv1", s, z, apply_relu=False
        )

        s_cat1, z_cat1 = inference.get_concat_quantization_params(
            inference.activation_ranges, "upconv1", "conv8"
        )
        M0_up1, shift_up1 = inference.compute_requantize_multiplier(s_up1, s_cat1)
        M0_e42, shift_e42 = inference.compute_requantize_multiplier(s_e42, s_cat1)

        q_x_aligned = inference.requantize_tensor(
            q_x, z_up1, z_cat1, M0_up1, shift_up1
        )
        q_skip_aligned = inference.requantize_tensor(
            q_e42, z_e42, z_cat1, M0_e42, shift_e42
        )

        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_cat, cfg["conv11"], "conv11", s_cat1, z_cat1, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv12"], "conv12", s, z, apply_relu=True
        )

        # Decoder block 2 (skip from conv6)
        q_x, s_up2, z_up2, _, _, _ = inference.run_integer_layer(
            q_x, cfg["upconv2"], "upconv2", s, z, apply_relu=False
        )

        s_cat2, z_cat2 = inference.get_concat_quantization_params(
            inference.activation_ranges, "upconv2", "conv6"
        )
        M0_up2, shift_up2 = inference.compute_requantize_multiplier(s_up2, s_cat2)
        M0_e32, shift_e32 = inference.compute_requantize_multiplier(s_e32, s_cat2)

        q_x_aligned = inference.requantize_tensor(
            q_x, z_up2, z_cat2, M0_up2, shift_up2
        )
        q_skip_aligned = inference.requantize_tensor(
            q_e32, z_e32, z_cat2, M0_e32, shift_e32
        )

        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_cat, cfg["conv13"], "conv13", s_cat2, z_cat2, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv14"], "conv14", s, z, apply_relu=True
        )

        # Decoder block 3 (skip from conv4)
        q_x, s_up3, z_up3, _, _, _ = inference.run_integer_layer(
            q_x, cfg["upconv3"], "upconv3", s, z, apply_relu=False
        )

        s_cat3, z_cat3 = inference.get_concat_quantization_params(
            inference.activation_ranges, "upconv3", "conv4"
        )
        M0_up3, shift_up3 = inference.compute_requantize_multiplier(s_up3, s_cat3)
        M0_e22, shift_e22 = inference.compute_requantize_multiplier(s_e22, s_cat3)

        q_x_aligned = inference.requantize_tensor(
            q_x, z_up3, z_cat3, M0_up3, shift_up3
        )
        q_skip_aligned = inference.requantize_tensor(
            q_e22, z_e22, z_cat3, M0_e22, shift_e22
        )

        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_cat, cfg["conv15"], "conv15", s_cat3, z_cat3, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv16"], "conv16", s, z, apply_relu=True
        )

        # Decoder block 4 (skip from conv2)
        q_x, s_up4, z_up4, _, _, _ = inference.run_integer_layer(
            q_x, cfg["upconv4"], "upconv4", s, z, apply_relu=False
        )

        s_cat4, z_cat4 = inference.get_concat_quantization_params(
            inference.activation_ranges, "upconv4", "conv2"
        )
        M0_up4, shift_up4 = inference.compute_requantize_multiplier(s_up4, s_cat4)
        M0_e12, shift_e12 = inference.compute_requantize_multiplier(s_e12, s_cat4)

        q_x_aligned = inference.requantize_tensor(
            q_x, z_up4, z_cat4, M0_up4, shift_up4
        )
        q_skip_aligned = inference.requantize_tensor(
            q_e12, z_e12, z_cat4, M0_e12, shift_e12
        )

        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_cat, cfg["conv17"], "conv17", s_cat4, z_cat4, apply_relu=True
        )
        q_x, s, z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["conv18"], "conv18", s, z, apply_relu=True
        )

        # Final output conv (no ReLU)
        q_out, final_s, final_z, _, _, _ = inference.run_integer_layer(
            q_x, cfg["outconv"], "outconv", s, z, apply_relu=False
        )

        # Dequantize final logits and update confusion counts
        with torch.no_grad():
            q_out_float = q_out.to(torch.float32)
            dequantized_logits = final_s * (q_out_float - final_z)

            int_probs = torch.sigmoid(dequantized_logits)
            int_preds = (int_probs > 0.5).float()

            preds_flat = int_preds.view(-1)
            masks_flat = masks.view(-1)

            tp += (preds_flat * masks_flat).sum().item()
            tn += ((1 - preds_flat) * (1 - masks_flat)).sum().item()
            fp += (preds_flat * (1 - masks_flat)).sum().item()
            fn += ((1 - preds_flat) * masks_flat).sum().item()

        seen += images.size(0)
        remaining = max(target_samples - seen, 0)
        print(
            f"[bench][{dataset_name}][integer] Batch {batch_idx}: processed {seen}/{target_samples} samples, remaining {remaining}."
        )

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    return {"dice": dice, "iou": iou, "acc": acc, "f1": f1}


def benchmark(bench: str = "all", num_data: Optional[int] = None) -> dict:
    """Run float and integer evaluation for selected dataset(s).

    If num_data is provided, only the first num_data samples from each
    test split are used for a quicker benchmark.
    """

    _disable_inference_debug_trace()

    supported_datasets = ("Skin-Lesion", "Flood", "Brain-MRI-Seg", "BUSI")
    if bench == "all":
        benchmark_datasets = supported_datasets
    else:
        benchmark_datasets = (bench,)

    results = {}
    for dataset_name in benchmark_datasets:
        loader = _get_test_loader(dataset_name)
        
        model = _build_model(dataset_name)

        float_metrics = _float_metrics(model, loader, dataset_name, num_data=num_data)
        int_metrics = _integer_metrics(model, loader, dataset_name, num_data=num_data)

        results[dataset_name] = {
            "float": float_metrics,
            "integer": int_metrics,
        }

    return results


def _get_results_filename(bench: str) -> str:
    """Return output json filename for the selected benchmark target."""
    filename_map = {
        "BUSI": "benchmark_results_busi.json",
        "Brain-MRI-Seg": "benchmark_results_brain_mri_seg.json",
        "Skin-Lesion": "benchmark_results_skin_lesion.json",
        "Flood": "benchmark_results_flood.json",
    }
    return filename_map.get(bench, "benchmark_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench",
        type=str,
        default="all",
        choices=["all", "Skin-Lesion", "Flood", "Brain-MRI-Seg", "BUSI"],
        help="Dataset to benchmark. Use 'all' to run every dataset.",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=None,
        help="Number of test samples to benchmark per dataset. If omitted, benchmarks the full test split.",
    )

    args = parser.parse_args()

    metrics = benchmark(bench=args.bench, num_data=args.num_data)
    results_file = _get_results_filename(args.bench)
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {results_file} with:")
    for ds, vals in metrics.items():
        fl = vals["float"]
        it = vals["integer"]
        print(
            f"  {ds}: float(dice={fl['dice']:.4f}, iou={fl['iou']:.4f}, acc={fl['acc']:.4f}, f1={fl['f1']:.4f}), "
            f"integer(dice={it['dice']:.4f}, iou={it['iou']:.4f}, acc={it['acc']:.4f}, f1={it['f1']:.4f})"
        )
