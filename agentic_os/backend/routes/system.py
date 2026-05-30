from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agentic_os.backend import config
from agentic_os.backend.sources import folders, git_activity

router = APIRouter(prefix="/api")


@router.get("/folders")
def get_folders():
    return folders.registry()


@router.get("/activity")
def activity():
    try:
        return git_activity.recent_commits(config.REPO_ROOT, n=10)
    except Exception as e:
        return JSONResponse({"error": True, "detail": f"activity: {e}"})
