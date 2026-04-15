"""Benchmark float vs fixed-point inference on Skin-Lesion and Flood.

Runs the same models and Dynamic Fixed-Point pipeline as inference.py over the
corresponding 10% test splits (from u_net.setup_data) and saves accuracies to
benchmark_results.json in the form:

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
from utils import get_fractional_bits, quantize_fixed_point, dequantize_fixed_point


def _disable_inference_debug_trace() -> None:
    """Turn off heavy debug logging in inference when benchmarking.

    This prevents run_fixed_point_layer/pool_fixed_point from storing full
    tensors in inference.debug_trace while we iterate over the test set.
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

    # Accumulate confusion counts across all batches
    tp = tn = fp = fn = 0.0
    seen = 0

    # Determine how many samples we expect to process
    total_samples: Optional[int]
    if hasattr(loader, "dataset"):
        total_samples = len(loader.dataset)
    else:
        total_samples = None
    if num_data is not None and total_samples is not None:
        total_samples = min(total_samples, num_data)
    elif num_data is not None:
        total_samples = num_data

    print(f"[float][{dataset_name}] starting benchmark; total samples: {total_samples or 'unknown'}")

    with torch.no_grad():
        for images, masks in loader:
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

            if total_samples is not None:
                remaining = max(total_samples - seen, 0)
                print(
                    f"[float][{dataset_name}] processed {seen}/{total_samples} samples (remaining {remaining})"
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
    """Compute average fixed-point Dice, IoU, accuracy, F1 using UNet path.

    This mirrors the Dynamic Fixed-Point path in inference.main, but runs over
    all batches from the 10% test DataLoader.
    """
    tp = tn = fp = fn = 0.0
    seen = 0

    # Determine how many samples we expect to process
    total_samples: Optional[int]
    if hasattr(loader, "dataset"):
        total_samples = len(loader.dataset)
    else:
        total_samples = None
    if num_data is not None and total_samples is not None:
        total_samples = min(total_samples, num_data)
    elif num_data is not None:
        total_samples = num_data

    print(f"[fixed-point][{dataset_name}] starting benchmark; total samples: {total_samples or 'unknown'}")

    for images, masks in loader:
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

        # 2) Quantize input using conv1 input range (Dynamic Fixed-Point)
        in_abs_max = inference.activation_ranges["conv1"]["in_abs_max"]
        pseudo_in_tensor = torch.tensor([in_abs_max])
        f_in = get_fractional_bits(pseudo_in_tensor, num_bits=8)
        q_x = quantize_fixed_point(images, f_in, dtype=torch.int8)

        # 3) Fixed-Point forward pass (mirrors inference.main UNet path)
        cfg = inference._get_layer_config(model)

        # Encoder block 1
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv1"], "conv1", f_in, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv2"], "conv2", f_out, apply_relu=True)
        q_e12, f_e12 = q_x, f_out
        q_x = inference.pool_fixed_point(q_x, name="pool1")

        # Encoder block 2
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv3"], "conv3", f_out, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv4"], "conv4", f_out, apply_relu=True)
        q_e22, f_e22 = q_x, f_out
        q_x = inference.pool_fixed_point(q_x, name="pool2")

        # Encoder block 3
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv5"], "conv5", f_out, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv6"], "conv6", f_out, apply_relu=True)
        q_e32, f_e32 = q_x, f_out
        q_x = inference.pool_fixed_point(q_x, name="pool3")

        # Encoder block 4
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv7"], "conv7", f_out, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv8"], "conv8", f_out, apply_relu=True)
        q_e42, f_e42 = q_x, f_out
        q_x = inference.pool_fixed_point(q_x, name="pool4")

        # Bottleneck
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv9"], "conv9", f_out, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv10"], "conv10", f_out, apply_relu=True)

        # Decoder block 1 (skip from conv8)
        q_x, f_up1, _, _ = inference.run_fixed_point_layer(q_x, cfg["upconv1"], "upconv1", f_out, apply_relu=False)
        f_cat1 = inference.get_concat_fractional_bits(inference.activation_ranges, "upconv1", "conv8")
        q_x_aligned = inference.align_tensor_for_concat(q_x, f_up1, f_cat1)
        q_skip_aligned = inference.align_tensor_for_concat(q_e42, f_e42, f_cat1)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_cat, cfg["conv11"], "conv11", f_cat1, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv12"], "conv12", f_out, apply_relu=True)

        # Decoder block 2 (skip from conv6)
        q_x, f_up2, _, _ = inference.run_fixed_point_layer(q_x, cfg["upconv2"], "upconv2", f_out, apply_relu=False)
        f_cat2 = inference.get_concat_fractional_bits(inference.activation_ranges, "upconv2", "conv6")
        q_x_aligned = inference.align_tensor_for_concat(q_x, f_up2, f_cat2)
        q_skip_aligned = inference.align_tensor_for_concat(q_e32, f_e32, f_cat2)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_cat, cfg["conv13"], "conv13", f_cat2, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv14"], "conv14", f_out, apply_relu=True)

        # Decoder block 3 (skip from conv4)
        q_x, f_up3, _, _ = inference.run_fixed_point_layer(q_x, cfg["upconv3"], "upconv3", f_out, apply_relu=False)
        f_cat3 = inference.get_concat_fractional_bits(inference.activation_ranges, "upconv3", "conv4")
        q_x_aligned = inference.align_tensor_for_concat(q_x, f_up3, f_cat3)
        q_skip_aligned = inference.align_tensor_for_concat(q_e22, f_e22, f_cat3)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_cat, cfg["conv15"], "conv15", f_cat3, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv16"], "conv16", f_out, apply_relu=True)

        # Decoder block 4 (skip from conv2)
        q_x, f_up4, _, _ = inference.run_fixed_point_layer(q_x, cfg["upconv4"], "upconv4", f_out, apply_relu=False)
        f_cat4 = inference.get_concat_fractional_bits(inference.activation_ranges, "upconv4", "conv2")
        q_x_aligned = inference.align_tensor_for_concat(q_x, f_up4, f_cat4)
        q_skip_aligned = inference.align_tensor_for_concat(q_e12, f_e12, f_cat4)
        q_cat = torch.cat([q_x_aligned, q_skip_aligned], dim=1)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_cat, cfg["conv17"], "conv17", f_cat4, apply_relu=True)
        q_x, f_out, _, _ = inference.run_fixed_point_layer(q_x, cfg["conv18"], "conv18", f_out, apply_relu=True)

        # Final output conv (no ReLU)
        outconv_f_in = f_out
        q_out, final_f_out, _, _ = inference.run_fixed_point_layer(
            q_x, cfg["outconv"], "outconv", outconv_f_in, apply_relu=False
        )

        # Dequantize final logits and update confusion counts
        with torch.no_grad():
            dequantized_logits = dequantize_fixed_point(q_out, final_f_out)

            int_probs = torch.sigmoid(dequantized_logits)
            int_preds = (int_probs > 0.5).float()

            preds_flat = int_preds.view(-1)
            masks_flat = masks.view(-1)

            tp += (preds_flat * masks_flat).sum().item()
            tn += ((1 - preds_flat) * (1 - masks_flat)).sum().item()
            fp += (preds_flat * (1 - masks_flat)).sum().item()
            fn += ((1 - preds_flat) * masks_flat).sum().item()

        seen += images.size(0)

        if total_samples is not None:
            remaining = max(total_samples - seen, 0)
            print(
                f"[fixed-point][{dataset_name}] processed {seen}/{total_samples} samples (remaining {remaining})"
            )

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)

    return {"dice": dice, "iou": iou, "acc": acc, "f1": f1}


def benchmark(num_data: Optional[int] = None) -> dict:
    """Run float and integer evaluation for Skin-Lesion and Flood.

    If num_data is provided, only the first num_data samples from each
    test split are used for a quicker benchmark.
    """

    _disable_inference_debug_trace()

    results = {}
    for dataset_name in ("Skin-Lesion", "Flood"):
        loader = _get_test_loader(dataset_name)
        
        model = _build_model(dataset_name)

        float_metrics = _float_metrics(model, loader, dataset_name, num_data=num_data)
        int_metrics = _integer_metrics(model, loader, dataset_name, num_data=num_data)

        results[dataset_name] = {
            "float": float_metrics,
            "integer": int_metrics,
        }

    return results


if __name__ == "__main__":
    metrics = benchmark()
    with open("benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved benchmark_results.json with:")
    for ds, vals in metrics.items():
        fl = vals["float"]
        it = vals["integer"]
        print(
            f"  {ds}: float(dice={fl['dice']:.4f}, iou={fl['iou']:.4f}, acc={fl['acc']:.4f}, f1={fl['f1']:.4f}), "
            f"integer(dice={it['dice']:.4f}, iou={it['iou']:.4f}, acc={it['acc']:.4f}, f1={it['f1']:.4f})"
        )