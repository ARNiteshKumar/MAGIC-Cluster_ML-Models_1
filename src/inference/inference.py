"""
Simple-BEV ONNX Runtime Inference
Usage: python src/inference/inference.py --model artifacts/simple_bev_optimized.onnx
"""
import argparse, time
import numpy as np
import onnxruntime as ort

def build_session(model_path, provider="CPUExecutionProvider"):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, opts, providers=[provider])
    print(f"✅ Loaded: {model_path}")
    return session

def benchmark(session, input_array, warmup=5, runs=50):
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    for _ in range(warmup): session.run([oname], {iname: input_array})
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run([oname], {iname: input_array})
        latencies.append((time.perf_counter() - t0) * 1000)
    lat = np.array(latencies)
    return {"mean_ms": float(lat.mean()), "p95_ms": float(np.percentile(lat,95)), "fps": float(1000/lat.mean())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/simple_bev_optimized.onnx")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--input_shape", nargs="+", type=int, default=[1,6,3,224,400])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    session = build_session(args.model, args.provider)
    dummy = np.random.randn(*args.input_shape).astype(np.float32)
    stats = benchmark(session, dummy, args.warmup, args.runs)
    print(f"\nMean: {stats['mean_ms']:.2f} ms  P95: {stats['p95_ms']:.2f} ms  FPS: {stats['fps']:.2f}")
    print("✅ Inference complete!")

if __name__ == "__main__": main()
