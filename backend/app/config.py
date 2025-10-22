from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Runtime configuration for the GAVID backend."""

    api_title: str = "GPU-Accelerated Vision Inference Dashboard"
    api_version: str = "0.1.0"
    api_description: str = (
        "FastAPI backend for GPU-accelerated image inference using PyTorch/TensorRT "
        "with CuPy preprocessing and NVIDIA DCGM telemetry."
    )

    model_repo: str = "resnet18"
    model_fp16: bool = True
    model_engine_path: Path = Path("artifacts/resnet18_fp16_engine.tsrt")
    model_torch_weights: Optional[str] = None
    device: str = "cuda:0"

    metrics_update_interval_s: float = 2.0
    dcgm_field_ids: tuple[int, ...] = (
        100,  # DCGM_FI_DEV_GPU_UTIL
        101,  # DCGM_FI_DEV_MEM_COPY_UTIL
        203,  # DCGM_FI_DEV_FB_USED
        204,  # DCGM_FI_DEV_FB_TOTAL
        150,  # DCGM_FI_DEV_POWER_USAGE
        155,  # DCGM_FI_DEV_POWER_LIMIT
        232,  # DCGM_FI_DEV_GPU_TEMP
    )

    allow_cpu_fallback: bool = True
    max_batch_size: int = 8
    enable_profiling: bool = True

    class Config:
        env_prefix = "GAVID_"
        case_sensitive = False

    @validator("model_engine_path", pre=True)
    def _expand_engine_path(cls, raw: str | os.PathLike[str]) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path


settings = Settings()
