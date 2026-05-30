from __future__ import annotations

import shutil
import time
from pathlib import Path

DONE_CELL = "✅ Done"


def _is_experiment_done(text: str, eid: str) -> bool:
    for ln in text.splitlines():
        if ln.strip().startswith(f"| {eid} "):
            return DONE_CELL in ln
    return False


def _edit_line(text: str, eid: str) -> str:
    lines = text.splitlines(keepends=True)
    found = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f"| {eid} "):
            cells = ln.rstrip("\n").split("|")
            # A well-formed 5-column body row splits into exactly 7 elements:
            # ['', ' eid ', ' script ', ' what ', ' status ', ' notes ', '']
            # (leading + trailing empty strings from the surrounding pipes).
            if len(cells) != 7:
                raise ValueError(
                    f"Malformed experiment row for {eid}: expected 5 columns, "
                    f"got {len(cells) - 2} (split produced {len(cells)} elements)"
                )
            # status is the 4th data column -> index 4 in split (leading empty at 0)
            cells[4] = f" {DONE_CELL} "
            newline = "|".join(cells)
            if ln.endswith("\n"):
                newline += "\n"
            lines[i] = newline
            found = True
            break
    if not found:
        raise ValueError(f"Experiment id not found in table: {eid}")
    return "".join(lines)


def mark_experiment_done(path: Path, eid: str, dry_run: bool = False) -> None:
    """Surgically flip one experiment row to Done; backup + read-back verify."""
    original = path.read_text()
    updated = _edit_line(original, eid)   # raises ValueError if id missing

    if dry_run:
        return

    backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, backup)

    path.write_text(updated)

    # Read-back verify: re-open and re-parse the persisted file.
    persisted = path.read_text()
    if not _is_experiment_done(persisted, eid):
        shutil.copy2(backup, path)   # restore
        raise RuntimeError(
            f"read-back verification failed for exp {eid}; restored from {backup}"
        )
    # Success: remove backup so .bak files don't accumulate next to memory files.
    backup.unlink(missing_ok=True)
