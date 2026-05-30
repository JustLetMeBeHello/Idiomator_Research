from __future__ import annotations

import re

SYSTEMS = ["A", "B", "B4", "C", "C4", "D", "E", "F", "G"]


def _kv_pairs(line: str) -> dict[str, float]:
    """Parse 'D=0.7515, A=0.7486, ...' into {label: value}."""
    out = {}
    for m in re.finditer(r"(B4|C4|[A-G])=([0-9.]+)", line):
        out[m.group(1)] = float(m.group(2))
    return out


def _section(text: str, header_substr: str) -> str:
    """Return the first non-empty content line under a header containing header_substr."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("##") and header_substr in ln:
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt
    raise ValueError(f"Section not found: {header_substr!r}")


def _parse_cls_line(line: str) -> dict[str, float]:
    """Parse 'A/D: 0.7823 overall; E: 0.7760; ...' (A/D share a value)."""
    out = {}
    for chunk in line.split(";"):
        m = re.search(r"([A-G0-9/]+):\s*([0-9.]+)", chunk)
        if not m:
            continue
        val = float(m.group(2))
        for label in m.group(1).split("/"):
            out[label.strip()] = val
    return out


def parse_systems(text: str) -> dict[str, dict]:
    """Build {system_label: {joint_f1, stability, cls_f1}} from key_numbers.md text."""
    joint = _kv_pairs(_section(text, "Joint F1"))
    stab = _kv_pairs(_section(text, "Stability Score"))
    cls = _parse_cls_line(_section(text, "Classification F1"))
    out: dict[str, dict] = {}
    for s in SYSTEMS:
        out[s] = {
            "joint_f1": joint.get(s),
            "stability": stab.get(s),
            "cls_f1": cls.get(s),
        }
    return out


def parse_ablation(text: str) -> list[dict]:
    """Parse the 15-combo ablation markdown table into a list of row dicts."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("##") and "Ablation Matrix" in ln:
            start = i
            break
    if start is None:
        raise ValueError("Ablation Matrix section not found")
    rows = []
    for ln in lines[start:]:
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip().strip("*") for c in ln.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        if cells[0].lower() in ("combo", "") or set(cells[0]) <= {"-"}:
            continue
        try:
            rows.append({
                "combo": cells[0],
                "span_f1": float(cells[1]),
                "indo_f1": float(cells[2]),
                "stability": float(cells[3]),
            })
        except ValueError:
            continue
    if not rows:
        raise ValueError("Ablation Matrix table had no parseable rows")
    return rows
