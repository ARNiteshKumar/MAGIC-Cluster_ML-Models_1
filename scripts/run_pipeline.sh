#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
SKIP_TRAIN=false
for arg in "$@"; do [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true; done
START=$(date +%s)
echo -e "${CYAN}====================================${NC}"
echo -e "${CYAN}  Simple-BEV ONNX PRODUCTION PIPELINE${NC}"
echo -e "${CYAN}====================================${NC}"
echo -e "  Start: $(date)  |  SkipTrain: $SKIP_TRAIN\n"

echo -e "${YELLOW}[STEP 1/5] Environment Setup${NC}"
bash scripts/setup.sh

if [ "$SKIP_TRAIN" = false ]; then
    echo -e "${YELLOW}[STEP 2/5] Training Model${NC}"
    bash scripts/train.sh configs/config.yaml
else
    echo -e "${YELLOW}⏭️  Skipping training${NC}"
fi

echo -e "${YELLOW}[STEP 3/5] Ecporiting to ONNX${NC}"
bash scripts/export.sh artifacts/simple_bev.pt

echo -e "${YELLOW}[STEP 4/5] Running ONNX Inference${NC}"
bash scripts/infer.sh artifacts/simple_bev_optimized.onnx

echo -e "${YELLOW}[STEP 5/5] Benchmarking${NC}"
bash scripts/benchmark.sh artifacts/simple_bev_optimized.onnx 50

ELAPSED=$(($(date +%s) - START))
echo -e "\n${CYAL}=== Pipeline Complete ✅ (${ELAPSED}s) ===${NC}"
echo -e "  Model PT : artifacts/simple_bev.pt"
echo -e "  Model ONNX: artifacts/simple_bev_optimized.onnx"
echo -e "  Report   : artifacts/benchmark_results.txt"
