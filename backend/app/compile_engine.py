from __future__ import annotations

import argparse
import io
import numpy as np
from PIL import Image

from .config import settings
from .inference import ResNet18Engine


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compile TensorRT engines for GAVID.")
    parser.add_argument(
        "--engine-path",
        type=str,
        default=str(settings.model_engine_path),
        help="Destination for the serialized TensorRT engine.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=settings.max_batch_size,
        help="Maximum batch size for TensorRT optimization profile.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=settings.model_fp16,
        help="Compile engine with FP16 precision.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even if the engine already exists.",
    )
    return parser.parse_args()


def _dummy_image() -> bytes:
    array = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    engine = ResNet18Engine(
        engine_path=args.engine_path,
        fp16=args.fp16,
        max_batch_size=args.max_batch,
    )
    engine.predict(_dummy_image())
    print(f"TensorRT engine written to {args.engine_path}")


if __name__ == "__main__":
    main()
