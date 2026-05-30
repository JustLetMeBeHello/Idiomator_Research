from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agentic_os.backend import config
from agentic_os.backend.parsers import key_numbers, project_overview
from agentic_os.backend.sources import results, systems_view

router = APIRouter(prefix="/api/research")


def _err(detail: str):
    return JSONResponse({"error": True, "detail": detail})


@router.get("/systems")
def systems():
    try:
        text = config.KEY_NUMBERS_MD.read_text()
        curated = key_numbers.parse_systems(text)
        try:
            live = results.live_by_label(config.RESULTS_JSON)
        except FileNotFoundError:
            live = {}
        return systems_view.merge(curated, live)
    except Exception as e:  # fail loud, never invent
        return _err(f"systems: {e}")


@router.get("/ablation")
def ablation():
    try:
        return key_numbers.parse_ablation(config.KEY_NUMBERS_MD.read_text())
    except Exception as e:
        return _err(f"ablation: {e}")


@router.get("/experiments")
def experiments():
    try:
        return project_overview.parse_experiments(config.PROJECT_OVERVIEW_MD.read_text())
    except Exception as e:
        return _err(f"experiments: {e}")


@router.get("/blockers")
def blockers():
    try:
        return project_overview.parse_blockers(config.PROJECT_OVERVIEW_MD.read_text())
    except Exception as e:
        return _err(f"blockers: {e}")
