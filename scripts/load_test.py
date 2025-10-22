from __future__ import annotations

import argparse
import io
import random
import time

import requests
from PIL import Image


def random_image_bytes(size: int = 224) -> bytes:
    image = Image.new(
        "RGB",
        (size, size),
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire-and-forget load generator for GAVID.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/infer")
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()

    session = requests.Session()

    latencies = []
    for idx in range(args.requests):
        files = {
            "file": (
                f"sample-{idx}.png",
                random_image_bytes(),
                "image/png",
            )
        }
        start = time.perf_counter()
        response = session.post(args.url, files=files, timeout=30)
        response.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)

    print(f"Sent {len(latencies)} requests")
    print(f"Mean latency: {sum(latencies) / len(latencies):.2f} ms")
    print(f"P95 latency: {sorted(latencies)[int(0.95 * len(latencies))]:.2f} ms")


if __name__ == "__main__":
    main()
