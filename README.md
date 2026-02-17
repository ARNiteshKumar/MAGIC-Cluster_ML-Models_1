# 🚗 Simple-BEV ONNX Export Pipeline

> **Week 2 Assignment** — Optimize the Simple-BEV model, export to ONNX, and verify ONNX Runtime  
> Reference: [aharley/simple_bev](https://github.com/aharley/simple_bev)

---

## 📌 Overview

This project implements a **Bird's Eye View (BEV) perception model** for autonomous driving, exports it to ONNX format, optimizes it with ONNX Simplifier, and benchmarks it against PyTorch baseline.

| Item | Detail |
|------|--------|
| Model | SimpleBEV (custom implementation) |
| Input | `[B, 6, 3, 224, 400]` — 6 camera views |
| Output | `[B, 8, 200, 200]` — BEV segmentation map |
| Parameters | ~7.8M |
| ONNX Opset | 17 |
| Runtime | ONNX Runtime 1.24.1 |
| GPU (export) | Tesla T4 (Google Colab) |

---

## 📁 Directory Structure

```
simple-bev-onnx/
├── src/
│   ├── models/simple_bev.py        # Model architecture
│   ├── training/train.py           # Training script
│   ├── inference/inference.py      # ONNX Runtime inference
│   └── data/datacard.md            # Data card
├── scripts/
│   ├── run_pipeline.sh             # ⭐ Master: runs everything
│   ├── setup.sh                    # Install dependencies
│   ├── train.sh                    # Train model
│   ├── export.sh                   # Export to ONNX
│   ├── infer.sh                    # Run inference
│   └── benchmark.sh                # PyTorch vs ONNX comparison
├── configs/config.yaml             # Hyperparameters & paths
├── artifacts/benchmark_results.txt # Benchmark report
├── docs/deployment_guide.md        # Deployment guide
├── notebooks/                      # Colab notebook
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### ⭐ Option 1: Full Pipeline (One Command)
```bash
git clone https://github.com/ARNiteshKumar/MAGIC-Cluster_ML-Models_1
cd simple-bev-onnx
bash scripts/run_pipeline.sh
```
This runs: **Setup → Train → Export ONNX → Inference → Benchmark** automatically.

### Option 2: Step by Step
```bash
bash scripts/setup.sh       # 1. Install dependencies
bash scripts/train.sh       # 2. Train model
bash scripts/export.sh      # 3. Export to ONNX
bash scripts/infer.sh       # 4. Run inference
bash scripts/benchmark.sh   # 5. Benchmark report
```

### Skip Training (use pre-trained weights from Release)
```bash
bash scripts/run_pipeline.sh --skip-train
```

### Run in Docker
```bash
docker build -t simple-bev-onnx .
docker run --rm simple-bev-onnx
```

---

## 📊 Benchmark Results

Tested on **Google Colab T4 GPU** (ONNX Runtime on CPU for portability):

| Metric | PyTorch CPU | ONNX Runtime | Speedup |
|--------|------------|--------------|---------|
| Mean Latency | 504.02 ms | 393.70 ms | **1.28x** |
| P95 Latency | 664.65 ms | 563.65 ms | 1.18x |
| Throughput | 1.98 FPS | 2.54 FPS | — |
| Output Shape | [1,8,200,200] | [1,8,200,200] | ✅ Match |
| Numerical Diff | — | 1.49e-08 | ✅ Valid |

> 🚀 ONNX Runtime is **1.28x faster** than PyTorch on CPU  
> ✅ Numerical verification PASSED (max diff: 1.49e-08)

---

## 🧠 Model Architecture

```
SimpleBEVModel
├── BEVEncoder     (ResNet-style backbone)
├── BEVSplat       (Feature projection to BEV grid)
├── FusionLayer    (Fuses all 6 camera features)
└── BEVDecoder     (Upsample + segmentation head)
```

---

## 📦 Model Artifacts

> ⚠️ ONNX model files are tracked via [Git Releases](../../releases).  
> Re-generate locally by running the notebook.

| File | Description |
|------|-------------|
| `simple_bev.onnx` | Base ONNX export (opset 17) |
| `simple_bev_optimized.onnx` | onnxsim-optimized ONNX |
| `simple_bev.pt` | PyTorch state dict |

---

## 🔗 References

- [Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?](https://github.com/aharley/simple_bev)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)
