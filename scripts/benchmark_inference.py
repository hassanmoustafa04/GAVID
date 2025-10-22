from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from app.inference import ResNet18Engine  # noqa: E402


def random_image(size: int = 224) -> bytes:
    array = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_benchmark(device: str, iters: int, warmup: int) -> float:
    engine = ResNet18Engine(device=device)
    payload = random_image()

    for _ in range(warmup):
        engine.predict(payload)

    latencies = []
    for _ in range(iters):
        start = time.perf_counter()
        engine.predict(payload)
        latencies.append((time.perf_counter() - start) * 1000)

    return sum(latencies) / len(latencies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU and TensorRT inference latency.")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    cpu_latency = run_benchmark("cpu", args.iters, args.warmup)
    print(f"CPU mean latency: {cpu_latency:.2f} ms")

    if not os.environ.get("CUDA_VISIBLE_DEVICES", "0"):
        print("Skipping GPU benchmark; no CUDA devices visible.")
        return

    gpu_latency = run_benchmark("cuda:0", args.iters, args.warmup)
    print(f"GPU (TensorRT) mean latency: {gpu_latency:.2f} ms")
    if gpu_latency > 0:
        print(f"Speedup vs CPU: {cpu_latency / gpu_latency:.2f}×")


if __name__ == "__main__":
    main()
