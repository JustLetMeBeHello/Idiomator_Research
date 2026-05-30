from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get(
    "AOS_REPO_ROOT",
    "/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training",
)).resolve()

MEMORY_DIR = Path(os.environ.get(
    "AOS_MEMORY_DIR",
    "/Users/shishirmaddineni/.claude/projects/"
    "-Users-shishirmaddineni-Desktop-Idiomator-Research-Research-And-Training/memory",
)).resolve()

RESULTS_JSON = REPO_ROOT / "results" / "pipeline_eval" / "pipeline_eval_results.json"
KEY_NUMBERS_MD = MEMORY_DIR / "key_numbers.md"
PROJECT_OVERVIEW_MD = MEMORY_DIR / "project_overview.md"


def validate() -> None:
    """Fail loud at startup if any required source path is missing."""
    if not REPO_ROOT.is_dir():
        raise RuntimeError(f"REPO_ROOT not found: {REPO_ROOT}")
    if not MEMORY_DIR.is_dir():
        raise RuntimeError(f"MEMORY_DIR not found: {MEMORY_DIR}")
    if not KEY_NUMBERS_MD.is_file():
        raise RuntimeError(f"key_numbers.md not found: {KEY_NUMBERS_MD}")
    if not PROJECT_OVERVIEW_MD.is_file():
        raise RuntimeError(f"project_overview.md not found: {PROJECT_OVERVIEW_MD}")
