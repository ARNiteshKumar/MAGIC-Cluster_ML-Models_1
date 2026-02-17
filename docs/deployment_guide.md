# Deployment Guide — Simple-BEV ONNX

## Option 1: Direct Python
```bash
pip install -r requirements.txt
python src/inference/inference.py --model artifacts/simple_bev_optimized.onnx
```

## Option 2: Bash Pipeline
```bash
bash scripts/run_pipeline.sh
```

## Option 3: Docker
```bash
docker build -t simple-bev-onnx .
docker run --rm simple-bev-onnx
```

## Expected Performance
| Environment | Mean Latency | FPS |
|-------------|-------------|-----|
| CPU RORT) | ~394 ms | ~2.5 |
| CPU (PyTorch) | ~504 ms | ~2.0 |
| T4 GPU (PyTorch) | ~12 ms | ~86 |

## Troubleshooting
- Model not found: run `bash scripts/export.sh` or the Colab notebook
- Slow CPU: set `export OMP_NUM_THREADS=4`
