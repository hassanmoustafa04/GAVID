from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .inference import ResNet18Engine, get_engine
from .metrics import collector
from .schema import GPUUtilizationResponse, InferenceResponse


def create_app(model_engine: ResNet18Engine | None = None) -> FastAPI:
    if model_engine is None:
        model_engine = get_engine()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/infer", response_model=InferenceResponse)
    async def infer(file: UploadFile = File(...)) -> InferenceResponse:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            result = model_engine.predict(payload)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result

    @app.get("/metrics/gpu", response_model=GPUUtilizationResponse)
    async def gpu_metrics(samples: int = Query(default=1, ge=1, le=120)) -> GPUUtilizationResponse:
        return collector.collect(num_samples=samples)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        collector.close()

    # Serve frontend static files (must be last so API routes take precedence)
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")

    return app


app = create_app()
