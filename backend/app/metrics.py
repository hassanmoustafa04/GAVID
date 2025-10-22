from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
except ImportError:  # pragma: no cover - executed only without NVML
    pynvml = None  # type: ignore

try:
    from pydcgm import DcgmSystem  # type: ignore
except ImportError:  # pragma: no cover - executed only without DCGM
    DcgmSystem = None  # type: ignore

from .config import settings
from .schema import GPUUtilizationResponse, GPUUtilizationSample


@dataclass
class _MetricSample:
    gpu_util: float
    mem_util: float
    fb_used_mb: float
    fb_total_mb: float
    power_w: float
    power_limit_w: float
    temperature_c: float
    timestamp: float


class _MetricBackend:
    def sample(self) -> _MetricSample:  # pragma: no cover - interface only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        return None

    @property
    def device_name(self) -> Optional[str]:
        return None

    @property
    def source(self) -> str:
        return "unknown"


class _DCGMBackend(_MetricBackend):
    """Use NVIDIA DCGM for high-fidelity GPU telemetry."""

    def __init__(self) -> None:
        if DcgmSystem is None:
            raise RuntimeError("pydcgm not installed.")
        self.system = DcgmSystem()
        self.system.discovery.inject_field_ids(settings.dcgm_field_ids)
        self.system.discovery.update()
        self._device_ids = self.system.discovery.gpus()
        if not self._device_ids:
            raise RuntimeError("No GPU devices discovered via DCGM.")
        self._device_id = self._device_ids[0]
        self.system.watch_fields(settings.dcgm_field_ids, update_freq=settings.metrics_update_interval_s)

    def sample(self) -> _MetricSample:
        values = self.system.values(self._device_id)
        field_map = {field: value for field, value in values.items()}

        def _get(field_id: int, default: float = 0.0) -> float:
            return float(field_map.get(field_id, default))

        total = _get(204, 1.0) / 1024  # convert MiB to GiB placeholder
        return _MetricSample(
            gpu_util=_get(100),
            mem_util=_get(101),
            fb_used_mb=_get(203),
            fb_total_mb=_get(204),
            power_w=_get(150) / 1000.0,
            power_limit_w=_get(155) / 1000.0,
            temperature_c=_get(232),
            timestamp=time.time(),
        )

    def close(self) -> None:
        self.system.unwatch_fields(settings.dcgm_field_ids)

    @property
    def source(self) -> str:
        return "dcgm"

    @property
    def device_name(self) -> Optional[str]:
        return f"GPU {self._device_id}"


class _NVMLBackend(_MetricBackend):
    """Fallback metrics provider leveraging NVIDIA Management Library."""

    def __init__(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml not available.")
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample(self) -> _MetricSample:
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self._handle)
        limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)

        return _MetricSample(
            gpu_util=float(util.gpu),
            mem_util=float(util.memory),
            fb_used_mb=mem.used / (1024 * 1024),
            fb_total_mb=mem.total / (1024 * 1024),
            power_w=power / 1000.0,
            power_limit_w=limit / 1000.0,
            temperature_c=float(temp),
            timestamp=time.time(),
        )

    def close(self) -> None:
        pynvml.nvmlShutdown()

    @property
    def source(self) -> str:
        return "nvml"

    @property
    def device_name(self) -> Optional[str]:
        return pynvml.nvmlDeviceGetName(self._handle).decode("utf-8")


class _StubBackend(_MetricBackend):
    """Return static values when neither DCGM nor NVML is accessible."""

    def sample(self) -> _MetricSample:
        now = time.time()
        # Provide deterministic dummy metrics with slight oscillation.
        phase = (now % 10) / 10.0
        util = 20.0 + 5.0 * (1 + phase)
        return _MetricSample(
            gpu_util=util,
            mem_util=util / 2,
            fb_used_mb=1024.0,
            fb_total_mb=8192.0,
            power_w=150.0 + 10.0 * phase,
            power_limit_w=250.0,
            temperature_c=45.0 + 2.0 * phase,
            timestamp=now,
        )

    @property
    def source(self) -> str:
        return "stub"

    @property
    def device_name(self) -> Optional[str]:
        return "virtual-gpu"


class MetricsCollector:
    """Expose a unified interface for GPU telemetry sampling."""

    def __init__(self) -> None:
        self._backend = self._initialize_backend()

    def _initialize_backend(self) -> _MetricBackend:
        for backend_cls in (_DCGMBackend, _NVMLBackend):
            try:
                return backend_cls()
            except Exception:
                continue
        return _StubBackend()

    def collect(self, num_samples: int = 1) -> GPUUtilizationResponse:
        num_samples = max(1, num_samples)
        samples = []
        for idx in range(num_samples):
            samples.append(self._backend.sample())
            if idx < num_samples - 1:
                time.sleep(settings.metrics_update_interval_s)
        payload = GPUUtilizationResponse(
            samples=[
                GPUUtilizationSample(
                    gpu_util=sample.gpu_util,
                    mem_util=sample.mem_util,
                    memory_used_mb=sample.fb_used_mb,
                    memory_total_mb=sample.fb_total_mb,
                    power_w=sample.power_w,
                    power_limit_w=sample.power_limit_w,
                    temperature_c=sample.temperature_c,
                    timestamp=sample.timestamp,
                )
                for sample in samples
            ],
            source=self._backend.source,
            interval_s=settings.metrics_update_interval_s,
            device=self._backend.device_name,
        )
        return payload

    def close(self) -> None:
        self._backend.close()


collector = MetricsCollector()
