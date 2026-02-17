#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
MODEL=${1:-artifacts/simple_bev_optimized.onnx}
PROVIDER=${2:-CPUExecutionProvider}
[ ! -f "$MODEL" ] && MODEL="artifacts/simple_bev.onnx"
echo -e "${GREEN}=== Simple-BEV -- ONNX Inference ($MODEL) ===${NC}"
python3 src/inference/inference.py --model "$MODEL" --provider "$PROVIDER" --input_shape 1 6 3 224 400 --warmup 5 --runs 50
echo -e "${GREEN}✅ Inference complete!${NC}"
