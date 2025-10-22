from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.schema import (
    ClassificationCandidate,
    GPUUtilizationResponse,
    GPUUtilizationSample,
    InferenceResponse,
)


class _FakeEngine:
    def predict(self, _: bytes) -> InferenceResponse:
        return InferenceResponse(
            top1=ClassificationCandidate(label="cat", confidence=0.42),
            top5=[
                ClassificationCandidate(label="cat", confidence=0.42),
                ClassificationCandidate(label="dog", confidence=0.33),
                ClassificationCandidate(label="fox", confidence=0.09),
                ClassificationCandidate(label="wolf", confidence=0.08),
                ClassificationCandidate(label="horse", confidence=0.08),
            ],
            latency_ms=12.5,
            throughput_fps=80.0,
            engine="stub",
            batch_size=1,
        )


class _FakeCollector:
    def collect(self, num_samples: int = 1) -> GPUUtilizationResponse:
        return GPUUtilizationResponse(
            samples=[
                GPUUtilizationSample(
                    gpu_util=50.0,
                    mem_util=25.0,
                    memory_used_mb=2048.0,
                    memory_total_mb=8192.0,
                    power_w=150.0,
                    power_limit_w=250.0,
                    temperature_c=55.0,
                    timestamp=0.0,
                )
                for _ in range(num_samples)
            ],
            source="stub",
            interval_s=1.0,
            device="fake-gpu",
        )


def test_health_endpoint() -> None:
    app = create_app(model_engine=_FakeEngine())
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inference_endpoint(monkeypatch) -> None:
    app = create_app(model_engine=_FakeEngine())
    client = TestClient(app)
    response = client.post("/infer", files={"file": ("test.png", b"abcd", "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["top1"]["label"] == "cat"
    assert payload["engine"] == "stub"


def test_metrics_endpoint(monkeypatch) -> None:
    app = create_app(model_engine=_FakeEngine())
    import app.main as main_module

    monkeypatch.setattr(main_module, "collector", _FakeCollector())
    client = TestClient(app)
    response = client.get("/metrics/gpu?samples=2")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "stub"
    assert len(payload["samples"]) == 2
