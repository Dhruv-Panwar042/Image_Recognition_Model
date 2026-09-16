import sys
from pathlib import Path

# Ensure backend directory is first in sys.path so 'app' package resolves cleanly
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.v1.endpoints import router as api_v1_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
## Multimodal Computer Vision & Visual Intelligence REST API

This API provides a decoupled, production-ready microservice architecture that combines:
* **YOLOv8 Nano**: Real-time object detection across 80 classes with execution latency breakdown.
* **Spatial Distance Estimation**: Normalized centroid calculations and proximity categorization.
* **Density Heatmaps**: Matrix confidence accumulation with Gaussian smoothing and JET colormapping.
* **Google Gemini 2.5 Flash**: Multimodal scene understanding, structured public safety risk assessments, and visual Q&A.
* **Enterprise Reporting**: Programmatic PDF report generation and structured JSON audit exports.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Latency & SLA Profiling Middleware ──────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time)
    return response

# ── CORS Middleware ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount API Routers ───────────────────────────────────────────
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# ── Mount Modern Frontend ───────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
index_path = frontend_dir / "index.html"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/health", tags=["System"])
def health_check():
    """Liveness & health check endpoint for container orchestrators (Railway / Render)."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }

@app.get("/", include_in_schema=False)
def root():
    """Serves the modern enterprise dashboard UI directly at root URL."""
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Visit /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
