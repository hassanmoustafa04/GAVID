from __future__ import annotations

import io
import numpy as np
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image, UnidentifiedImageError

try:
    import cupy as cp
    from cupy import cuda
except ImportError:  # pragma: no cover - executed only without CuPy
    cp = None  # type: ignore
    cuda = None  # type: ignore

try:
    import torch_tensorrt
except ImportError:  # pragma: no cover - executed only without TensorRT
    torch_tensorrt = None  # type: ignore

from torch import nn
from torch.utils import dlpack
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from .config import settings
from .schema import ClassificationCandidate, InferenceResponse


class CuPyPreprocessor:
    """CuPy accelerated preprocessing pipeline for ResNet-style inputs."""

    def __init__(self, size: int = 224) -> None:
        if cp is None:
            raise RuntimeError("CuPy is required for CuPyPreprocessor but was not found.")
        self.size = size
        self.mean = cp.asarray([0.485, 0.456, 0.406], dtype=cp.float32)[:, None, None]
        self.std = cp.asarray([0.229, 0.224, 0.225], dtype=cp.float32)[:, None, None]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Resize on CPU using Pillow for best fidelity before GPU transfer.
        resized = image.convert("RGB").resize((self.size, self.size))
        cpu_array = np.asarray(resized, dtype=np.float32)
        np_img = cp.ascontiguousarray(cp.asarray(cpu_array))
        np_img = cp.transpose(np_img, (2, 0, 1))  # HWC -> CHW
        np_img /= 255.0
        np_img = (np_img - self.mean) / self.std
        gpu_tensor = dlpack.from_dlpack(np_img.toDlpack())
        return gpu_tensor


class TorchPreprocessor:
    """Fallback CPU preprocessing that mimics torchvision defaults."""

    def __init__(self, size: int = 224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))


class ResNet18Engine:
    """TensorRT-accelerated inference engine with PyTorch fallback."""

    def __init__(
        self,
        engine_path: Path | str | None = None,
        device: str = settings.device,
        fp16: bool = settings.model_fp16,
        max_batch_size: int = settings.max_batch_size,
        allow_cpu_fallback: bool = settings.allow_cpu_fallback,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.allow_cpu_fallback = allow_cpu_fallback
        self.engine_path = Path(engine_path) if engine_path else settings.model_engine_path
        self.labels = ResNet18_Weights.DEFAULT.meta["categories"]

        self._preprocessor = self._select_preprocessor()
        self._torch_model: Optional[nn.Module] = None
        self._trt_module: Optional[nn.Module] = None

        self._load_models()

    def _select_preprocessor(self):
        if self.device.type == "cuda" and cp is not None:
            return CuPyPreprocessor()
        return TorchPreprocessor()

    def _load_models(self) -> None:
        weights = ResNet18_Weights.DEFAULT
        try:
            model = resnet18(weights=weights)
            self.labels = weights.meta["categories"]
        except Exception as exc:  # pragma: no cover - offline fallback
            model = resnet18(weights=None)
            if settings.model_torch_weights:
                state_dict = torch.load(settings.model_torch_weights, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                raise RuntimeError(
                    "Unable to load pretrained ResNet18 weights. "
                    "Set GAVID_MODEL_TORCH_WEIGHTS to a local .pth file."
                ) from exc
            self.labels = [f"class_{idx}" for idx in range(1000)]
        model.eval()
        if self.device.type == "cuda":
            model = model.to(self.device)
            if self.fp16:
                model = model.half()

        self._torch_model = model

        if self.device.type == "cuda" and torch_tensorrt is not None:
            self._trt_module = self._load_or_compile_trt(model)
        elif self.device.type != "cuda" and not self.allow_cpu_fallback:
            raise RuntimeError("CUDA device required but not available.")

    def _load_or_compile_trt(self, model: nn.Module) -> Optional[nn.Module]:
        engine_path = self.engine_path
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        if engine_path.exists():
            serialized = engine_path.read_bytes()
            buffer = io.BytesIO(serialized)
            module = torch.jit.load(buffer, map_location=self.device)  # type: ignore[arg-type]
            module.eval()
            return module

        if torch_tensorrt is None:
            return None

        if hasattr(torch_tensorrt, "Device"):
            device_spec = torch_tensorrt.Device("cuda:0")
        else:  # pragma: no cover - Torch-TensorRT legacy fallback
            device_spec = torch.device("cuda:0")

        compile_inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(self.max_batch_size // 2 + 1, 3, 224, 224),
                max_shape=(self.max_batch_size, 3, 224, 224),
                dtype=torch.half if self.fp16 else torch.float,
                device=device_spec,
            )
        ]

        enabled_precisions: Iterable[torch.dtype] = (
            {torch.float, torch.half} if self.fp16 else {torch.float}
        )

        trt_module = torch_tensorrt.compile(
            model,
            inputs=compile_inputs,
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
        )

        buffer = io.BytesIO()
        torch.jit.save(trt_module, buffer)
        buffer.seek(0)
        engine_path.write_bytes(buffer.read())
        return trt_module

    def _ensure_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda":
            tensor = tensor.to(self.device)
            if self.fp16:
                tensor = tensor.half()
        return tensor

    def predict(self, image_bytes: bytes) -> InferenceResponse:
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except UnidentifiedImageError as exc:
            raise ValueError("Provided file is not a valid image.") from exc

        preprocess_start = time.perf_counter()
        inputs = self._preprocessor(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        tensor = self._ensure_batch(inputs)
        tensor = self._to_device(tensor)

        exec_start = time.perf_counter()
        with torch.no_grad():
            if self._trt_module is not None:
                outputs = self._trt_module(tensor)
                engine_name = "TensorRT"
            else:
                assert self._torch_model is not None
                outputs = self._torch_model(tensor)
                engine_name = "PyTorch"
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        exec_ms = (time.perf_counter() - exec_start) * 1000

        scores = torch.softmax(outputs[0], dim=0).cpu()
        top_scores, top_idxs = torch.topk(scores, k=5)
        top_candidates = [
            ClassificationCandidate(label=self.labels[idx], confidence=float(score))
            for score, idx in zip(top_scores, top_idxs)
        ]

        latency_ms = preprocess_ms + exec_ms
        throughput = 1000.0 / latency_ms if latency_ms > 0 else float("inf")

        return InferenceResponse(
            top1=top_candidates[0],
            top5=top_candidates,
            latency_ms=latency_ms,
            throughput_fps=throughput,
            engine=engine_name,
            batch_size=tensor.shape[0],
        )


_ENGINE: Optional[ResNet18Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> ResNet18Engine:
    """Return a singleton instance of the inference engine."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = ResNet18Engine(engine_path=settings.model_engine_path)
    return _ENGINE
