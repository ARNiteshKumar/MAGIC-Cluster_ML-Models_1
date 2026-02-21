"""
Simple-BEV Inference & Evaluation Script
=========================================
Runs both PyTorch and ONNX inference on nuScenes-style data.
Outputs:
  - BEV segmentation maps with bounding boxes per class
  - Per-class and mean IoU (Intersection over Union)
  - Per-class and overall accuracy
  - MSC verification (PyTorch vs ONNX numerical comparison)
  - Latency benchmarks for both backends
  - Saved bbox images into output folder

Usage:
  python src/inference/inference.py --config configs/config.yaml
  python src/inference/inference.py --config configs/config.yaml --num_samples 16
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================= HELPERS =====================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CLASS_NAMES = [
    "background", "drivable_surface", "vehicle", "pedestrian",
    "cyclist", "road_marking", "static_obstacle", "other",
]

CLASS_COLORS = np.array([
    [0, 0, 0],        # background
    [128, 128, 128],   # drivable_surface
    [255, 0, 0],       # vehicle
    [0, 255, 0],       # pedestrian
    [0, 0, 255],       # cyclist
    [255, 255, 0],     # road_marking
    [255, 165, 0],     # static_obstacle
    [128, 0, 128],     # other
], dtype=np.uint8)


def _load_class_info(cfg: dict):
    """Override class names/colors from config if present."""
    global CLASS_NAMES, CLASS_COLORS
    if "classes" in cfg:
        if "names" in cfg["classes"]:
            CLASS_NAMES = cfg["classes"]["names"]
        if "colors" in cfg["classes"]:
            CLASS_COLORS = np.array(cfg["classes"]["colors"], dtype=np.uint8)


# ============================= DATA LOADER =================================

def get_synthetic_dataset(cfg: dict, n_samples: int = 64):
    """Generate synthetic nuScenes-style multi-camera images and BEV labels."""
    inp = cfg["input"]
    mod = cfg["model"]
    imgs = np.random.randn(n_samples, mod["ncams"], inp["channels"],
                           inp["height"], inp["width"]).astype(np.float32)
    labels = np.random.randint(0, mod["num_classes"],
                               (n_samples, mod["bev_h"], mod["bev_w"])).astype(np.int64)
    return imgs, labels


# ===================== BOUNDING BOX EXTRACTION =============================

def extract_bboxes_from_segmentation(pred_map: np.ndarray, num_classes: int,
                                     min_area: int = 10):
    """
    Extract bounding boxes from a BEV segmentation map.

    Args:
        pred_map: (H, W) int array with class indices
        num_classes: total number of classes
        min_area: minimum connected-component area to keep

    Returns:
        list of dicts with keys: class_id, class_name, bbox (x, y, w, h), area
    """
    detections = []
    for cls_id in range(1, num_classes):  # skip background (0)
        mask = (pred_map == cls_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        for comp_id in range(1, num_labels):  # skip background component
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x = stats[comp_id, cv2.CC_STAT_LEFT]
            y = stats[comp_id, cv2.CC_STAT_TOP]
            w = stats[comp_id, cv2.CC_STAT_WIDTH]
            h = stats[comp_id, cv2.CC_STAT_HEIGHT]
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                "bbox": (x, y, w, h),
                "area": int(area),
                "centroid": (float(centroids[comp_id][0]),
                             float(centroids[comp_id][1])),
            })
    return detections


# ===================== VISUALISATION =======================================

def draw_bev_with_bboxes(pred_map: np.ndarray, detections: list,
                         title: str = "BEV Prediction"):
    """
    Create a coloured BEV segmentation image with bounding boxes overlaid.

    Returns:
        fig: matplotlib Figure
    """
    H, W = pred_map.shape
    rgb = CLASS_COLORS[pred_map]  # (H, W, 3)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14)

    for det in detections:
        x, y, w, h = det["bbox"]
        cls_id = det["class_id"]
        color = CLASS_COLORS[cls_id].astype(float) / 255.0
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none",
            boxstyle="round,pad=0.5")
        ax.add_patch(rect)
        ax.text(x, max(y - 2, 0),
                f"{det['class_name']} ({det['area']}px)",
                fontsize=7, color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=1))

    # legend
    handles = []
    for i, name in enumerate(CLASS_NAMES):
        if i == 0:
            continue
        handles.append(mpatches.Patch(
            color=CLASS_COLORS[i].astype(float) / 255.0, label=name))
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ===================== IoU / ACCURACY METRICS ==============================

def compute_iou_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    """
    Compute per-class IoU between prediction and ground truth.

    Args:
        pred: (N, H, W) predicted class indices
        gt:   (N, H, W) ground-truth class indices
        num_classes: number of classes

    Returns:
        iou_per_class: dict {class_name: iou}
        mean_iou: float
    """
    iou_per_class = {}
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, gt == cls).sum()
        union = np.logical_or(pred == cls, gt == cls).sum()
        if union == 0:
            iou = float("nan")
        else:
            iou = float(intersection) / float(union)
            ious.append(iou)
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        iou_per_class[name] = iou
    mean_iou = float(np.nanmean(ious)) if ious else 0.0
    return iou_per_class, mean_iou


def compute_accuracy(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    """
    Compute overall and per-class pixel accuracy.

    Returns:
        overall_acc: float
        per_class_acc: dict {class_name: accuracy}
    """
    overall_acc = float((pred == gt).sum()) / float(pred.size)
    per_class_acc = {}
    for cls in range(num_classes):
        mask = gt == cls
        if mask.sum() == 0:
            acc = float("nan")
        else:
            acc = float((pred[mask] == cls).sum()) / float(mask.sum())
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        per_class_acc[name] = acc
    return overall_acc, per_class_acc


# ===================== MSC VERIFICATION ====================================

def msc_verification(pytorch_output: np.ndarray, onnx_output: np.ndarray):
    """
    Model Similarity Check (MSC) between PyTorch and ONNX outputs.

    Compares raw logits (before argmax) to verify numerical consistency.

    Returns:
        dict with max_diff, mean_diff, cosine_similarity,
             pred_agreement (%), status
    """
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # cosine similarity (flatten both)
    a = pytorch_output.flatten().astype(np.float64)
    b = onnx_output.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = float(dot / (norm_a * norm_b + 1e-12))

    # prediction-level agreement (argmax match %)
    pt_pred = pytorch_output.argmax(axis=1)   # (B, H, W)
    ox_pred = onnx_output.argmax(axis=1)
    pred_agreement = float((pt_pred == ox_pred).sum()) / float(pt_pred.size) * 100.0

    status = "PASSED" if max_diff < 1e-4 and pred_agreement > 99.0 else "REVIEW"

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_similarity": cosine_sim,
        "prediction_agreement_pct": pred_agreement,
        "status": status,
    }


# ===================== PYTORCH INFERENCE ===================================

def pytorch_inference(imgs_np: np.ndarray, cfg: dict):
    """
    Run PyTorch inference.

    Args:
        imgs_np: (N, 6, 3, H, W) float32 numpy array
        cfg: config dict

    Returns:
        logits: (N, num_classes, bev_h, bev_w) float32 numpy
        latency_ms: mean latency per sample
    """
    import torch
    from src.models.simple_bev import build_model

    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    model.eval()

    # load checkpoint if available
    ckpt_path = cfg["inference"].get("pytorch_model_path", "artifacts/simple_bev.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded PyTorch checkpoint: {ckpt_path}")
    else:
        print(f"  No checkpoint found at {ckpt_path}, using random weights")

    imgs_t = torch.from_numpy(imgs_np).to(device)
    all_logits = []
    latencies = []

    with torch.no_grad():
        # warmup
        _ = model(imgs_t[:1])

        for i in range(imgs_np.shape[0]):
            t0 = time.perf_counter()
            out = model(imgs_t[i : i + 1])          # (1, C, H, W)
            latencies.append((time.perf_counter() - t0) * 1000)
            all_logits.append(out.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)  # (N, C, H, W)
    mean_lat = float(np.mean(latencies))
    return logits, mean_lat


# ===================== ONNX INFERENCE ======================================

def onnx_inference(imgs_np: np.ndarray, cfg: dict):
    """
    Run ONNX Runtime inference.

    Args:
        imgs_np: (N, 6, 3, H, W) float32 numpy array
        cfg: config dict

    Returns:
        logits: (N, num_classes, bev_h, bev_w) float32 numpy
        latency_ms: mean latency per sample
    """
    import onnxruntime as ort

    provider = cfg["inference"].get("provider", "CPUExecutionProvider")
    model_path = cfg["inference"].get("model_path",
                                      "artifacts/simple_bev_optimized.onnx")
    if not os.path.exists(model_path):
        # fallback to non-optimized
        model_path = model_path.replace("_optimized", "")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model not found. Looked for: {cfg['inference']['model_path']} "
            f"and {model_path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, opts, providers=[provider])
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    print(f"  Loaded ONNX model: {model_path}  (provider={provider})")

    all_logits = []
    latencies = []

    # warmup
    session.run([oname], {iname: imgs_np[:1]})

    for i in range(imgs_np.shape[0]):
        t0 = time.perf_counter()
        out = session.run([oname], {iname: imgs_np[i : i + 1]})
        latencies.append((time.perf_counter() - t0) * 1000)
        all_logits.append(out[0])

    logits = np.concatenate(all_logits, axis=0)
    mean_lat = float(np.mean(latencies))
    return logits, mean_lat


# ===================== REPORT GENERATION ===================================

def generate_report(pytorch_metrics: dict, onnx_metrics: dict,
                    msc: dict, cfg: dict, output_dir: str):
    """Write a comprehensive evaluation report to a text file."""
    num_classes = cfg["model"]["num_classes"]
    lines = []
    lines.append("=" * 70)
    lines.append("  SIMPLE-BEV INFERENCE & EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # ----- Class Information -----
    lines.append("CLASS INFORMATION")
    lines.append("-" * 40)
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        c = CLASS_COLORS[i] if i < len(CLASS_COLORS) else [0, 0, 0]
        lines.append(f"  {i}: {name:<20s}  RGB({c[0]:3d},{c[1]:3d},{c[2]:3d})")
    lines.append("")

    # ----- PyTorch Metrics -----
    lines.append("PYTORCH EVALUATION METRICS")
    lines.append("-" * 40)
    lines.append(f"  Mean latency      : {pytorch_metrics['latency_ms']:.2f} ms/sample")
    lines.append(f"  Overall accuracy  : {pytorch_metrics['overall_accuracy'] * 100:.2f}%")
    lines.append(f"  Mean IoU (mIoU)   : {pytorch_metrics['mean_iou'] * 100:.2f}%")
    lines.append("  Per-class IoU:")
    for name, val in pytorch_metrics["iou_per_class"].items():
        v = f"{val * 100:.2f}%" if not np.isnan(val) else "N/A"
        lines.append(f"    {name:<20s}: {v}")
    lines.append("  Per-class accuracy:")
    for name, val in pytorch_metrics["per_class_accuracy"].items():
        v = f"{val * 100:.2f}%" if not np.isnan(val) else "N/A"
        lines.append(f"    {name:<20s}: {v}")
    lines.append(f"  Total bboxes detected: {pytorch_metrics['total_bboxes']}")
    lines.append("")

    # ----- ONNX Metrics -----
    lines.append("ONNX EVALUATION METRICS")
    lines.append("-" * 40)
    lines.append(f"  Mean latency      : {onnx_metrics['latency_ms']:.2f} ms/sample")
    lines.append(f"  Overall accuracy  : {onnx_metrics['overall_accuracy'] * 100:.2f}%")
    lines.append(f"  Mean IoU (mIoU)   : {onnx_metrics['mean_iou'] * 100:.2f}%")
    lines.append("  Per-class IoU:")
    for name, val in onnx_metrics["iou_per_class"].items():
        v = f"{val * 100:.2f}%" if not np.isnan(val) else "N/A"
        lines.append(f"    {name:<20s}: {v}")
    lines.append("  Per-class accuracy:")
    for name, val in onnx_metrics["per_class_accuracy"].items():
        v = f"{val * 100:.2f}%" if not np.isnan(val) else "N/A"
        lines.append(f"    {name:<20s}: {v}")
    lines.append(f"  Total bboxes detected: {onnx_metrics['total_bboxes']}")
    lines.append("")

    # ----- MSC Verification -----
    lines.append("MSC VERIFICATION (PyTorch vs ONNX)")
    lines.append("-" * 40)
    lines.append(f"  Max absolute diff      : {msc['max_diff']:.2e}")
    lines.append(f"  Mean absolute diff     : {msc['mean_diff']:.2e}")
    lines.append(f"  Cosine similarity      : {msc['cosine_similarity']:.8f}")
    lines.append(f"  Prediction agreement   : {msc['prediction_agreement_pct']:.2f}%")
    lines.append(f"  Status                 : {msc['status']}")
    lines.append("")

    # ----- Latency Comparison -----
    pt_lat = pytorch_metrics["latency_ms"]
    ox_lat = onnx_metrics["latency_ms"]
    speedup = pt_lat / ox_lat if ox_lat > 0 else 0
    lines.append("LATENCY COMPARISON")
    lines.append("-" * 40)
    lines.append(f"  PyTorch mean latency   : {pt_lat:.2f} ms  ({1000/pt_lat:.2f} FPS)")
    lines.append(f"  ONNX RT mean latency   : {ox_lat:.2f} ms  ({1000/ox_lat:.2f} FPS)")
    lines.append(f"  Speedup (ONNX/PyTorch) : {speedup:.2f}x")
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  Output images saved to : {output_dir}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    return report, report_path


# ===================== MAIN ================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simple-BEV Inference with bbox output, IoU metrics & MSC verification")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of synthetic samples (overrides config)")
    parser.add_argument("--provider", default=None,
                        help="ONNX Runtime provider (overrides config)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _load_class_info(cfg)

    num_classes = cfg["model"]["num_classes"]
    num_samples = (args.num_samples
                   or cfg.get("evaluation", {}).get("num_synthetic_samples", 64))
    output_dir = (args.output_dir
                  or cfg.get("inference", {}).get("output_dir", "output_bev_results/"))
    min_area = cfg.get("inference", {}).get("min_component_area", 10)

    if args.provider:
        cfg["inference"]["provider"] = args.provider

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pytorch"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "onnx"), exist_ok=True)

    print("=" * 60)
    print("  Simple-BEV Inference & Evaluation")
    print("=" * 60)

    # ---- 1. Generate common input ----
    print(f"\n[1/5] Generating {num_samples} synthetic nuScenes-style samples ...")
    imgs, gt_labels = get_synthetic_dataset(cfg, n_samples=num_samples)
    print(f"  Input shape  : {imgs.shape}  (N, cams, C, H, W)")
    print(f"  Labels shape : {gt_labels.shape}  (N, bev_H, bev_W)")

    # ---- 2. PyTorch inference ----
    print("\n[2/5] Running PyTorch inference ...")
    pt_logits, pt_latency = pytorch_inference(imgs, cfg)
    pt_preds = pt_logits.argmax(axis=1)  # (N, H, W)
    print(f"  Output shape : {pt_logits.shape}")
    print(f"  Mean latency : {pt_latency:.2f} ms/sample")

    # ---- 3. ONNX inference ----
    print("\n[3/5] Running ONNX Runtime inference ...")
    ox_logits, ox_latency = onnx_inference(imgs, cfg)
    ox_preds = ox_logits.argmax(axis=1)  # (N, H, W)
    print(f"  Output shape : {ox_logits.shape}")
    print(f"  Mean latency : {ox_latency:.2f} ms/sample")

    # ---- 4. Compute metrics ----
    print("\n[4/5] Computing evaluation metrics ...")

    # MSC verification
    msc = msc_verification(pt_logits, ox_logits)
    print(f"  MSC status             : {msc['status']}")
    print(f"  Max diff               : {msc['max_diff']:.2e}")
    print(f"  Prediction agreement   : {msc['prediction_agreement_pct']:.2f}%")

    # PyTorch metrics vs ground truth
    pt_iou_per_class, pt_miou = compute_iou_per_class(pt_preds, gt_labels, num_classes)
    pt_overall_acc, pt_class_acc = compute_accuracy(pt_preds, gt_labels, num_classes)

    # ONNX metrics vs ground truth
    ox_iou_per_class, ox_miou = compute_iou_per_class(ox_preds, gt_labels, num_classes)
    ox_overall_acc, ox_class_acc = compute_accuracy(ox_preds, gt_labels, num_classes)

    print(f"\n  PyTorch  =>  mIoU: {pt_miou*100:.2f}%  Accuracy: {pt_overall_acc*100:.2f}%")
    print(f"  ONNX RT  =>  mIoU: {ox_miou*100:.2f}%  Accuracy: {ox_overall_acc*100:.2f}%")

    # ---- 5. Extract bboxes and save images ----
    print(f"\n[5/5] Extracting bounding boxes & saving visualisations to {output_dir}/ ...")

    pt_total_bboxes = 0
    ox_total_bboxes = 0

    max_save = min(num_samples, 20)  # save at most 20 images
    for i in range(max_save):
        # PyTorch bboxes
        pt_dets = extract_bboxes_from_segmentation(pt_preds[i], num_classes, min_area)
        pt_total_bboxes += len(pt_dets)
        fig_pt = draw_bev_with_bboxes(
            pt_preds[i], pt_dets,
            title=f"PyTorch BEV #{i}  ({len(pt_dets)} detections)")
        fig_pt.savefig(os.path.join(output_dir, "pytorch", f"bev_bbox_{i:04d}.png"),
                       dpi=120, bbox_inches="tight")
        plt.close(fig_pt)

        # ONNX bboxes
        ox_dets = extract_bboxes_from_segmentation(ox_preds[i], num_classes, min_area)
        ox_total_bboxes += len(ox_dets)
        fig_ox = draw_bev_with_bboxes(
            ox_preds[i], ox_dets,
            title=f"ONNX BEV #{i}  ({len(ox_dets)} detections)")
        fig_ox.savefig(os.path.join(output_dir, "onnx", f"bev_bbox_{i:04d}.png"),
                       dpi=120, bbox_inches="tight")
        plt.close(fig_ox)

    # count remaining bboxes for samples not saved as images
    for i in range(max_save, num_samples):
        pt_total_bboxes += len(
            extract_bboxes_from_segmentation(pt_preds[i], num_classes, min_area))
        ox_total_bboxes += len(
            extract_bboxes_from_segmentation(ox_preds[i], num_classes, min_area))

    print(f"  PyTorch total bboxes : {pt_total_bboxes}")
    print(f"  ONNX total bboxes    : {ox_total_bboxes}")
    print(f"  Images saved         : {max_save} per backend")

    # ---- Build metrics dicts ----
    pytorch_metrics = {
        "latency_ms": pt_latency,
        "overall_accuracy": pt_overall_acc,
        "mean_iou": pt_miou,
        "iou_per_class": pt_iou_per_class,
        "per_class_accuracy": pt_class_acc,
        "total_bboxes": pt_total_bboxes,
    }
    onnx_metrics = {
        "latency_ms": ox_latency,
        "overall_accuracy": ox_overall_acc,
        "mean_iou": ox_miou,
        "iou_per_class": ox_iou_per_class,
        "per_class_accuracy": ox_class_acc,
        "total_bboxes": ox_total_bboxes,
    }

    # ---- Generate report ----
    report, report_path = generate_report(pytorch_metrics, onnx_metrics,
                                          msc, cfg, output_dir)
    print(f"\n{'=' * 60}")
    print(report)
    print(f"\nReport saved to: {report_path}")
    print("Inference complete!")


if __name__ == "__main__":
    main()
