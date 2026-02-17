#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
MODEL=${1:-artifacts/simple_bev_optimized.onnx}; RUNS=${2:-50}
echo -e "${GREEN}=== Benchmark: PyTorch CPU vs ONNX Runtime ===${NC}"
python3 - <<EOF
import torch, onnxruntime as ort, numpy as np, time, sys, os
sys.path.insert(0,'.')
from src.models.simple_bev import SimpleBEVModel
model = SimpleBEVModel().eval(); dummy = torch.randn(1,6,3,224,400); dummy_np = dummy.numpy()
with torch.no_grad():
    for _ in range(5): model(dummy)
pt_lats=[]
with torch.no_grad():
    for _ in range(${RUNS}): t0=time.perf_counter(); model(dummy); pt_lats.append((time.perf_counter()-t0)*1000)
pt=np.array(pt_lats)
opts=ort.SessionOptions(); opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess=ort.InferenceSession('${MODEL}',providers=['CPUExecutionProvider'])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
ort_lats=[]
for _ in range(${RUNS}): t0=time.perf_counter(); sess.run([oname],{iname:dummy_np}); ort_lats.append((time.perf_counter()-t0)*1000)
ort=np.array(ort_lats); speedup=pt.mean()/ort.mean()
print(f"PyTorch CPU: {pt.mean():.2f}ms | ONNX RT: {ort.mean():.2f}ms | Speedup: {speedup:.2f}x")
EOF
echo -e "${GREEN}✅ Benchmark complete!${NC}"
