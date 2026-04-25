"""Benchmark float vs fixed-point inference on Skin-Lesion and Flood.

Runs the same models and Static 64-bit Fixed-Point pipeline as inference.py
over the corresponding 10% test splits (from u_net.setup_data) and saves accuracies to
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

import unet.FixedPoint64.inference as inference
import u_net
from unet.FixedPoint64.utils import quantize_fixed_point, dequantize_fixed_point


def _disable_inference_debug_trace() -> None:
    """Turn off heavy debug logging in inference when benchmarking.

    This prevents run_static_fixed_point_layer/pool_fixed_point from storing full
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
    """Compute average fixed-point Dice, IoU, accuracy, F1 using static UNet path.

    This mirrors the Static 64-bit Fixed-Point path in inference.main, but runs
    over all batches from the 10% test DataLoader.
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

        # 1) Quantize input directly to static Q31.32 int64
        q_x = quantize_fixed_point(images)

        # 2) Fixed-point forward pass (mirrors inference.main UNet static path)
        cfg = inference._get_layer_config(model)

        # Encoder block 1
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv1"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv2"], apply_relu=True)
        q_e12 = q_x
        q_x = inference.pool_fixed_point(q_x)

        # Encoder block 2
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv3"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv4"], apply_relu=True)
        q_e22 = q_x
        q_x = inference.pool_fixed_point(q_x)

        # Encoder block 3
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv5"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv6"], apply_relu=True)
        q_e32 = q_x
        q_x = inference.pool_fixed_point(q_x)

        # Encoder block 4
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv7"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv8"], apply_relu=True)
        q_e42 = q_x
        q_x = inference.pool_fixed_point(q_x)

        # Bottleneck
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv9"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv10"], apply_relu=True)

        # Decoder block 1 (skip from conv8)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["upconv1"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e42], dim=1)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_cat, cfg["conv11"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv12"], apply_relu=True)

        # Decoder block 2 (skip from conv6)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["upconv2"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e32], dim=1)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_cat, cfg["conv13"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv14"], apply_relu=True)

        # Decoder block 3 (skip from conv4)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["upconv3"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e22], dim=1)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_cat, cfg["conv15"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv16"], apply_relu=True)

        # Decoder block 4 (skip from conv2)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["upconv4"], apply_relu=False)
        q_cat = torch.cat([q_x, q_e12], dim=1)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_cat, cfg["conv17"], apply_relu=True)
        q_x, _, _ = inference.run_static_fixed_point_layer(q_x, cfg["conv18"], apply_relu=True)

        # Final output conv (no ReLU)
        q_out, _, _ = inference.run_static_fixed_point_layer(
            q_x, cfg["outconv"], apply_relu=False
        )

        # Dequantize final logits and update confusion counts
        with torch.no_grad():
            dequantized_logits = dequantize_fixed_point(q_out)

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
            "fixed-point": int_metrics,
        }

    return results


if __name__ == "__main__":
    metrics = benchmark()
    with open("benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved benchmark_results.json with:")
    for ds, vals in metrics.items():
        fl = vals["float"]
        it = vals["fixed-point"]
        print(
            f"  {ds}: float(dice={fl['dice']:.4f}, iou={fl['iou']:.4f}, acc={fl['acc']:.4f}, f1={fl['f1']:.4f}), "
            f"fixed-point(dice={it['dice']:.4f}, iou={it['iou']:.4f}, acc={it['acc']:.4f}, f1={it['f1']:.4f})"
        )