from __future__ import annotations

import re

STATUS_MAP = [
    ("SKIP", "skip"),
    ("Done", "done"),
    ("✅", "done"),
    ("Not planned", "not_planned"),
    ("Not run", "not_run"),
    ("❌", "not_run"),
]


def _status_of(cell: str) -> str:
    for needle, val in STATUS_MAP:
        if needle.lower() in cell.lower():
            return val
    return "unknown"


def parse_experiments(text: str) -> list[dict]:
    """Parse the rigor-experiments markdown table into row dicts."""
    out = []
    for ln in text.splitlines():
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 5 or not re.fullmatch(r"\d{2}", cells[0]):
            continue
        script = re.sub(r"[`*]", "", cells[1]).replace("TBD ", "").strip()
        out.append({
            "id": cells[0],
            "script": script,
            "what": re.sub(r"[`*]", "", cells[2]).strip(),
            "status": _status_of(cells[3]),
            "notes": cells[4],
        })
    if not out:
        raise ValueError("No experiment rows parsed from project_overview")
    return out


def parse_blockers(text: str) -> list[dict]:
    """Parse the numbered 'Blocking TODOs' lists from ALL sections."""
    out = []
    in_block = False
    for ln in text.splitlines():
        if "Blocking TODOs" in ln:
            in_block = True
            continue
        if in_block:
            if ln.strip().startswith("##"):
                # End this block but keep scanning for more sections
                in_block = False
                continue
            m = re.match(r"\d+\.\s+(.*)", ln.strip())
            if not m:
                continue
            body = m.group(1)
            title = re.sub(r"[*]", "", body).strip()
            out.append({
                "title": title,
                "hard_blocker": "HARD BLOCKER" in body,
            })
    if not out:
        raise ValueError("No blockers parsed from project_overview")
    return out
