"""VaultVision Backend - FastAPI Application Entry Point.

Vault Sync AI LLC - Baton Rouge, LA
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import CORS_ORIGINS, UPLOAD_DIR
from app.core.database import init_db
from app.api import videos, query, analytics, keys, websocket, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="VaultVision API",
    description="AI Video Intelligence Platform - Ctrl+F for Video",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files (thumbnails, etc.)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Register routers
app.include_router(videos.router)
app.include_router(query.router)
app.include_router(analytics.router)
app.include_router(keys.router)
app.include_router(websocket.router)
app.include_router(ingest.router)


@app.get("/")
async def root():
    return {
        "name": "VaultVision API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
