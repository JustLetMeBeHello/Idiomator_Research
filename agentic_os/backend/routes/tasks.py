from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agentic_os.backend import config
from agentic_os.backend.parsers import project_overview
from agentic_os.backend.tasks import derive, writeback

router = APIRouter(prefix="/api")


def _state():
    text = config.PROJECT_OVERVIEW_MD.read_text()
    exps = project_overview.parse_experiments(text)
    blockers = project_overview.parse_blockers(text)
    return exps, blockers


@router.get("/tasks")
def tasks():
    try:
        exps, blockers = _state()
        return derive.build_backlog(exps, blockers)
    except Exception as e:
        return JSONResponse({"error": True, "detail": f"tasks: {e}"})


@router.get("/next")
def next_step():
    try:
        exps, blockers = _state()
        return derive.next_step(exps, blockers)
    except Exception as e:
        return JSONResponse({"error": True, "detail": f"next: {e}"})


@router.post("/tasks/{task_id}/done")
def mark_done(task_id: str):
    try:
        if not task_id.startswith("exp-"):
            return JSONResponse(
                {"error": True, "detail": f"only experiment tasks writable: {task_id}"}
            )
        eid = task_id.removeprefix("exp-").replace("-dryrun", "")
        writeback.mark_experiment_done(config.PROJECT_OVERVIEW_MD, eid)
        exps, blockers = _state()
        return {"ok": True, "next": derive.next_step(exps, blockers)}
    except Exception as e:
        return JSONResponse({"error": True, "detail": f"done: {e}"})
