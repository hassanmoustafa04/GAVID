"""
GPU-Accelerated Vision Inference Dashboard backend package.

This package exposes a FastAPI application that delivers GPU-accelerated
image inference using PyTorch/TensorRT and CuPy preprocessing, along with
GPU telemetry endpoints backed by NVIDIA DCGM.
"""

from .inference import get_engine
from .main import create_app

__all__ = ["create_app", "get_engine"]
