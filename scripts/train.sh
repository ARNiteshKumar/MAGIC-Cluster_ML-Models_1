#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
CONFIG=${1:-configs/config.yaml}
echo -e "${GREEN}=== Simple-BEV -- Training (Config: $CONFIG) ===${NC}"
python3 src/training/train.py --config "$CONFIG"
echo -e "${GREEN}✅ Training complete! (artifacts/simple_bev.pt)${NC}"
