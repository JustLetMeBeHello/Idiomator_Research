from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_os.backend import config
from agentic_os.backend.routes import research, system

app = FastAPI(title="Personal Agentic OS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    config.validate()


app.include_router(system.router)
app.include_router(research.router)
