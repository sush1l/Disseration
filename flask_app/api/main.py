"""
main.py — FastAPI application entry point.

Run:
    uvicorn api.main:app --reload --port 8000

UI:   http://localhost:8000/
Docs: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION
from api.models.ml import registry
from api.routes import predict_router, symptoms_router

STATIC_DIR = Path(__file__).parent.parent / "static"


# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load all models once
    print("\n" + "═" * 55)
    print("  Loading ML artifacts...")
    print("═" * 55)
    registry.load_all()
    print("═" * 55)
    print("  API ready.\n")
    yield
    # Shutdown (nothing to clean up for in-memory models)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(predict_router)
app.include_router(symptoms_router)

# ─── Static files (UI) ────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Root → UI ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


# ─── Health / info ────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    return {
        "status":  "ok",
        "api":     API_TITLE,
        "version": API_VERSION,
        "docs":    "/docs",
    }


@app.get("/models/info", tags=["Models"], summary="Model metadata & performance")
async def model_info():
    return JSONResponse(content=registry.get_model_info())
