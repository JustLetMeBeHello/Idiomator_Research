from __future__ import annotations

import subprocess
from pathlib import Path


def recent_commits(repo_root: Path, n: int = 10) -> list[dict]:
    out = subprocess.run(
        ["git", "-C", str(repo_root), "log", f"-{n}", "--pretty=%h%x1f%s%x1f%cr"],
        capture_output=True, text=True, check=True,
    ).stdout
    commits = []
    for ln in out.splitlines():
        parts = ln.split("\x1f")
        if len(parts) == 3:
            commits.append({"hash": parts[0], "subject": parts[1], "when": parts[2]})
    return commits
