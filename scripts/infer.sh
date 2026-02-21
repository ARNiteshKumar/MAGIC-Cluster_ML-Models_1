#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

CONFIG=${1:-configs/config.yaml}
DATA_ROOT=${2:-}          # pass nuScenes path as 2nd arg, or empty for synthetic
VERSION=${3:-v1.0-mini}
SPLIT=${4:-mini_val}
OUTPUT_DIR=${5:-output_bev_results}

echo -e "${GREEN}=== Simple-BEV -- Full Dataset Validation ===${NC}"

if [ -n "$DATA_ROOT" ] && [ -d "$DATA_ROOT" ]; then
    echo -e "${YELLOW}Mode: nuScenes ($VERSION / $SPLIT)${NC}"
    echo -e "${YELLOW}Data: $DATA_ROOT${NC}"
    python3 src/inference/inference.py \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --version "$VERSION" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_DIR"
else
    echo -e "${YELLOW}Mode: Synthetic (no --data_root or path not found)${NC}"
    echo -e "${RED}TIP: bash scripts/infer.sh configs/config.yaml /path/to/nuscenes v1.0-trainval val${NC}"
    python3 src/inference/inference.py \
        --config "$CONFIG" \
        --synthetic \
        --output_dir "$OUTPUT_DIR"
fi

echo -e "${GREEN}Results saved to: $OUTPUT_DIR/${NC}"
