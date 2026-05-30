# Personal Agentic OS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local desktop-metaphor "Personal Agentic OS" — FastAPI backend over the research repo + React/vite shell with a fully-wired Research folder (6 panels) and a Next-Step engine that derives an ordered backlog from project state and write-back-verifies completions.

**Architecture:** FastAPI backend reads repo files + memory markdown fresh per request, exposes typed JSON. React/vite frontend renders a desktop with draggable windows. Next-Step engine derives tasks from `project_overview.md` ordering rules; completion does a surgical, backed-up, read-back-verified line edit on the source markdown.

**Tech Stack:** Python 3.12.7 (FastAPI, uvicorn, pytest), Node 26 (vite, React, TypeScript, vitest). Backend deps live in the existing `.venv` (already has fastapi). `from __future__ import annotations` is kept on every backend file as defensive forward-compat — harmless on 3.12.

**Reference spec:** `docs/superpowers/specs/2026-05-29-agentic-os-dashboard-design.md`

---

## Conventions for the implementing engineer

- **WORKING DIRECTORY (cwd for all work)** = the git worktree at
  `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training/.claude/worktrees/agentic-os-dashboard`.
  Create `agentic_os/` here. All `git add`/`git commit` happen here (branch `worktree-agentic-os-dashboard`).
- **PYTHON INTERPRETER (absolute)** = `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training/.venv/bin/python`
  (Python 3.12.7, fastapi already installed). The worktree has NO local `.venv`, so wherever this plan says
  `.venv/bin/python` or `.venv/bin/pip`, use this absolute path instead. Do NOT `source activate`; call the absolute path directly.
  Example test run: `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training/.venv/bin/python -m pytest agentic_os/backend/tests -v` (run from the worktree cwd; `PYTHONPATH=.` if imports fail).
- If `httpx` is missing (needed by FastAPI TestClient), install once:
  `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training/.venv/bin/pip install httpx`.
- **Memory/source-of-truth paths used by config.py** (absolute, stable):
  - Memory dir = `/Users/shishirmaddineni/.claude/projects/-Users-shishirmaddineni-Desktop-Idiomator-Research-Research-And-Training/memory`.
  - `config.REPO_ROOT` defaults to the stable **main checkout** root `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training` (survives worktree cleanup; holds canonical `results/pipeline_eval/pipeline_eval_results.json` + memory). Keep Task 1's `config.py` default exactly as written — do NOT point it at the worktree.
- Run backend live: `<abs-python> -m uvicorn agentic_os.backend.main:app --reload --port 8011` from the worktree cwd.
- Frontend lives in `agentic_os/frontend`; `npm install` then `npm run dev` (vite default port 5173). Tests: `npm run test`.
- **CLAUDE.md hard rules that bind this code:** never invent metric values (fail loud if a parse fails); never call a write "done" without re-reading the persisted artifact; report metrics at 2 decimals.
- **Python 3.9 compat (HARD):** target interpreter is 3.9.6. PEP 604 unions (`X | None`) are evaluated at function-def time on 3.9 and raise `TypeError`. Every backend `.py` file in this plan MUST start with `from __future__ import annotations` as its first line. With that import, all annotations (incl. `dict | None`, `list[dict]`, `dict[str, float]`) become lazy strings and are safe. The code blocks below show the import where a file uses `|`; add it to every backend file regardless.
- **Confirmed JSON shape** of `results/pipeline_eval/pipeline_eval_results.json` (probed from the real file): top-level dict keyed by 9 system ids (`system_a_mbert_pipeline`, `system_b_gpt_pipeline`, `system_b4_gpt_pipeline_4shot`, `system_c_gpt_single`, `system_c4_gpt_single_4shot`, `system_d_mbert_s1_joint_span`, `system_e_joint_end_to_end`, `system_f_sequential_phase1_ph2`, `system_g_bio_tagger`). Each value is **deeply nested**:
  - `cls_f1` → dict `{English, Hindi, Indonesian, Spanish, Telugu, Overall}` (floats). Overall cls F1 = `node["cls_f1"]["Overall"]`.
  - `joint_acc` → top-level float.
  - `joint_f1` → dict per language + `Overall`, and `node["joint_f1"]["Overall"]["macro_avg_f1"]` is the headline macro-avg Joint F1 (matches key_numbers; e.g. A = 0.7486).
  - `stability` → dict `{mean_joint, std_joint, worst_lang, worst_f1, gap, stability, langs}`; the scalar stability score = `node["stability"]["stability"]` (e.g. A = 0.7049).
  - `span_standalone` (A–F) / `span_all` (G) → `{exact:{...}, overlap:{...}}`.
  - `span_e2e`, `span_correct_id` → `{exact, overlap}`.
  - CI key: A–F use `small_lang_ci`, G uses `indonesian_ci` → `{n_examples, cls_accuracy, span_exact, span_overlap, joint_f1}` (each a 3-list `[point, lo, hi]`).
  System G uses key `span_all` (not `span_standalone`) and `indonesian_ci` (not `small_lang_ci`).

---

## File Structure

**Backend (`agentic_os/backend/`)**
- `__init__.py`, `config.py` — env-var paths, validated on import.
- `main.py` — FastAPI app, CORS, route registration.
- `sources/folders.py` — folder registry.
- `sources/results.py` — read pipeline_eval JSON.
- `sources/git_activity.py` — git log feed.
- `parsers/key_numbers.py` — parse memory/key_numbers.md tables.
- `parsers/project_overview.py` — parse exp queue + blockers.
- `tasks/derive.py` — build ordered backlog.
- `tasks/writeback.py` — backup + surgical edit + read-back verify.
- `routes/research.py`, `routes/tasks.py`, `routes/system.py` — endpoint groups.
- `tests/` — pytest + fixtures.

**Frontend (`agentic_os/frontend/src/`)**
- `api.ts` — typed fetch client.
- `os/Desktop.tsx`, `os/Window.tsx`, `os/Taskbar.tsx`, `os/IconGrid.tsx`, `os/NextStepBar.tsx`.
- `folders/research/{SystemsBoard,AblationMatrix,ExperimentQueue,Blockers,Activity,Tasks}.tsx`.
- `folders/{papers,agents,oscore}/StubRoom.tsx`.

---

## Task 1: Backend scaffold + config validation

**Files:**
- Create: `agentic_os/__init__.py` (empty)
- Create: `agentic_os/backend/__init__.py` (empty)
- Create: `agentic_os/backend/config.py`
- Test: `agentic_os/backend/tests/__init__.py` (empty), `agentic_os/backend/tests/test_config.py`

- [ ] **Step 1: Install backend deps**

Run: `.venv/bin/pip install fastapi "uvicorn[standard]" pytest httpx`
Expected: successful install, no errors.

- [ ] **Step 2: Write the failing test**

`agentic_os/backend/tests/test_config.py`:
```python
import pytest
from agentic_os.backend import config

def test_repo_root_exists():
    assert config.REPO_ROOT.is_dir()

def test_memory_dir_exists():
    assert config.MEMORY_DIR.is_dir()

def test_results_json_path_resolves():
    assert config.RESULTS_JSON.name == "pipeline_eval_results.json"

def test_validate_raises_on_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="MEMORY_DIR"):
        config.validate()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: agentic_os.backend.config`.

- [ ] **Step 4: Write config.py**

`agentic_os/backend/config.py`:
```python
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
```

Also create empty `agentic_os/__init__.py`, `agentic_os/backend/__init__.py`, `agentic_os/backend/tests/__init__.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add agentic_os/__init__.py agentic_os/backend/
git commit -m "feat(aos): backend config with startup path validation"
```

---

## Task 2: key_numbers.md parser — systems metrics

**Files:**
- Create: `agentic_os/backend/parsers/__init__.py` (empty)
- Create: `agentic_os/backend/parsers/key_numbers.py`
- Test: `agentic_os/backend/tests/test_key_numbers.py`
- Fixture: `agentic_os/backend/tests/fixtures/key_numbers_sample.md`

- [ ] **Step 1: Create the fixture**

`agentic_os/backend/tests/fixtures/key_numbers_sample.md` (real lines copied from memory/key_numbers.md):
```markdown
## IdiomBERT — Full EN+ES+HI+TE Training, Joint F1 (macro-avg)
D=0.7515, A=0.7486, F=0.7440, E=0.7381, C=0.6977, B4=0.6966, B=0.6915, C4=0.5118, G=0.4750

## Stability Score (mean − std, in-distribution languages)
D=0.7061, A=0.7049, F=0.6843, E=0.6774, C=0.6148, B4=0.6247, B=0.5999, C4=0.4442, G=0.4658

## Classification F1 — full training
A/D: 0.7823 overall; E: 0.7760; F: 0.7741; B: 0.7599; C: 0.7268; B4: 0.7278; C4: 0.6668
```

- [ ] **Step 2: Write the failing test**

`agentic_os/backend/tests/test_key_numbers.py`:
```python
from pathlib import Path
import pytest
from agentic_os.backend.parsers import key_numbers

FIX = Path(__file__).parent / "fixtures" / "key_numbers_sample.md"

def test_parse_joint_f1():
    out = key_numbers.parse_systems(FIX.read_text())
    assert out["D"]["joint_f1"] == 0.7515
    assert out["A"]["joint_f1"] == 0.7486
    assert out["G"]["joint_f1"] == 0.4750

def test_parse_stability():
    out = key_numbers.parse_systems(FIX.read_text())
    assert out["D"]["stability"] == 0.7061
    assert out["C4"]["stability"] == 0.4442

def test_parse_cls_f1_shared_ad():
    out = key_numbers.parse_systems(FIX.read_text())
    assert out["A"]["cls_f1"] == 0.7823
    assert out["D"]["cls_f1"] == 0.7823
    assert out["E"]["cls_f1"] == 0.7760

def test_missing_section_raises():
    with pytest.raises(ValueError, match="Joint F1"):
        key_numbers.parse_systems("# nothing useful here")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_key_numbers.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Write the parser**

`agentic_os/backend/parsers/key_numbers.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_key_numbers.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add agentic_os/backend/parsers/ agentic_os/backend/tests/test_key_numbers.py agentic_os/backend/tests/fixtures/key_numbers_sample.md
git commit -m "feat(aos): parse system metrics from key_numbers.md"
```

---

## Task 3: key_numbers.md parser — ablation matrix

**Files:**
- Modify: `agentic_os/backend/parsers/key_numbers.py`
- Modify: `agentic_os/backend/tests/test_key_numbers.py`
- Modify: `agentic_os/backend/tests/fixtures/key_numbers_sample.md`

- [ ] **Step 1: Append matrix rows to the fixture**

Append to `agentic_os/backend/tests/fixtures/key_numbers_sample.md`:
```markdown
## System G Language Ablation Matrix — E2E Span Overlap F1 (idiomatic, overall)

| Combo | E2E Span F1 | Indo held-out F1 | Stability (mean−std) |
|---|---|---|---|
| en | 0.7348 | 0.6474 | 0.4427 |
| en_es | 0.7824 | 0.6725 | 0.4522 |
| **en_es_hi_te** | **0.8072** | **0.7209** | **0.4699** |
```

- [ ] **Step 2: Add the failing test**

Append to `agentic_os/backend/tests/test_key_numbers.py`:
```python
def test_parse_ablation_matrix():
    rows = key_numbers.parse_ablation(FIX.read_text())
    by_combo = {r["combo"]: r for r in rows}
    assert by_combo["en"]["span_f1"] == 0.7348
    assert by_combo["en_es_hi_te"]["span_f1"] == 0.8072
    assert by_combo["en_es_hi_te"]["indo_f1"] == 0.7209
    assert by_combo["en_es_hi_te"]["stability"] == 0.4699

def test_ablation_missing_raises():
    with pytest.raises(ValueError, match="Ablation"):
        key_numbers.parse_ablation("no table here")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_key_numbers.py -k ablation -v`
Expected: FAIL — `parse_ablation` not defined.

- [ ] **Step 4: Add parse_ablation to key_numbers.py**

Append to `agentic_os/backend/parsers/key_numbers.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_key_numbers.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add agentic_os/backend/parsers/key_numbers.py agentic_os/backend/tests/test_key_numbers.py agentic_os/backend/tests/fixtures/key_numbers_sample.md
git commit -m "feat(aos): parse ablation matrix from key_numbers.md"
```

---

## Task 4: results.py — live JSON source + mismatch merge

**Files:**
- Create: `agentic_os/backend/sources/__init__.py` (empty)
- Create: `agentic_os/backend/sources/results.py`
- Test: `agentic_os/backend/tests/test_results.py`
- Fixture: `agentic_os/backend/tests/fixtures/pipeline_eval_sample.json`

- [ ] **Step 1: Create the JSON fixture**

`agentic_os/backend/tests/fixtures/pipeline_eval_sample.json` (mirrors the REAL nested shape — trimmed to the keys the loader reads):
```json
{
  "system_a_mbert_pipeline": {
    "cls_f1": {"English": 0.8063, "Overall": 0.7823},
    "joint_acc": 0.6464,
    "joint_f1": {"English": {"macro_f1": 0.8023}, "Overall": {"macro_f1": 0.7631, "macro_avg_f1": 0.7486}},
    "span_standalone": {"exact": {"Overall": 0.6287}, "overlap": {"Overall": 0.8168}},
    "span_e2e": {"exact": {"Overall": 0.4926}, "overlap": {"Overall": 0.6571}},
    "span_correct_id": {"exact": {"Overall": 0.6409}, "overlap": {"Overall": 0.8549}},
    "stability": {"mean_joint": 0.7486, "std_joint": 0.0437, "stability": 0.7049,
                  "langs": {"English": 0.8023, "Spanish": 0.7309, "Hindi": 0.687, "Telugu": 0.774}},
    "small_lang_ci": {"n_examples": 124, "joint_f1": [0.7339, 0.6532, 0.8065]}
  },
  "system_g_bio_tagger": {
    "cls_f1": {"Overall": 0.0},
    "joint_acc": 0.5952,
    "joint_f1": {"Overall": {"macro_f1": 0.4668, "macro_avg_f1": 0.4750}},
    "span_all": {"exact": {"Overall": 0.59}, "overlap": {"Overall": 0.79}},
    "span_e2e": {"exact": {"Overall": 0.59}, "overlap": {"Overall": 0.79}},
    "span_correct_id": {"exact": {"Overall": 0.59}, "overlap": {"Overall": 0.79}},
    "stability": {"mean_joint": 0.4658, "std_joint": 0.0, "stability": 0.4658, "langs": {}},
    "indonesian_ci": {"n_examples": 325, "joint_f1": [0.7192, 0.6782, 0.7577]}
  }
}
```

- [ ] **Step 2: Write the failing test**

`agentic_os/backend/tests/test_results.py`:
```python
from pathlib import Path
import pytest
from agentic_os.backend.sources import results

FIX = Path(__file__).parent / "fixtures" / "pipeline_eval_sample.json"

def test_load_returns_systems():
    data = results.load_results(FIX)
    assert "system_a_mbert_pipeline" in data
    # joint_f1 is a nested dict in the real shape
    assert data["system_a_mbert_pipeline"]["joint_f1"]["Overall"]["macro_avg_f1"] == 0.7486

def test_label_for_system_id():
    assert results.label_for("system_a_mbert_pipeline") == "A"
    assert results.label_for("system_g_bio_tagger") == "G"
    assert results.label_for("system_b4_gpt_pipeline_4shot") == "B4"

def test_live_joint_f1_by_label():
    live = results.live_by_label(FIX)
    assert live["A"]["joint_f1"] == 0.7486
    assert live["A"]["stability"] == 0.7049
    assert live["A"]["cls_f1"] == 0.7823
    assert live["G"]["joint_f1"] == 0.4750

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        results.load_results(Path("/no/such.json"))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_results.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Write results.py**

`agentic_os/backend/sources/results.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

ID_TO_LABEL = {
    "system_a_mbert_pipeline": "A",
    "system_b_gpt_pipeline": "B",
    "system_b4_gpt_pipeline_4shot": "B4",
    "system_c_gpt_single": "C",
    "system_c4_gpt_single_4shot": "C4",
    "system_d_mbert_s1_joint_span": "D",
    "system_e_joint_end_to_end": "E",
    "system_f_sequential_phase1_ph2": "F",
    "system_g_bio_tagger": "G",
}


def label_for(system_id: str) -> str:
    if system_id not in ID_TO_LABEL:
        raise KeyError(f"Unknown system id: {system_id}")
    return ID_TO_LABEL[system_id]


def load_results(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"results json not found: {path}")
    return json.loads(path.read_text())


def _dig(node: dict, *path):
    """Walk nested dicts; return None if any key is missing/non-dict."""
    cur = node
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def live_by_label(path: Path) -> dict:
    """Reduce the full nested JSON to {label: {joint_f1, stability, cls_f1}} scalars."""
    raw = load_results(path)
    out = {}
    for sid, node in raw.items():
        label = ID_TO_LABEL.get(sid)
        if label is None:
            continue
        out[label] = {
            # headline macro-avg Joint F1 lives at joint_f1.Overall.macro_avg_f1
            "joint_f1": _dig(node, "joint_f1", "Overall", "macro_avg_f1"),
            # scalar stability score lives at stability.stability
            "stability": _dig(node, "stability", "stability"),
            # overall classification F1 lives at cls_f1.Overall
            "cls_f1": _dig(node, "cls_f1", "Overall"),
        }
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_results.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add agentic_os/backend/sources/ agentic_os/backend/tests/test_results.py agentic_os/backend/tests/fixtures/pipeline_eval_sample.json
git commit -m "feat(aos): live results JSON loader with label mapping"
```

---

## Task 5: merge curated + live with mismatch flag

**Files:**
- Create: `agentic_os/backend/sources/systems_view.py`
- Test: `agentic_os/backend/tests/test_systems_view.py`

- [ ] **Step 1: Write the failing test**

`agentic_os/backend/tests/test_systems_view.py`:
```python
from agentic_os.backend.sources import systems_view

def test_merge_prefers_curated_and_flags_mismatch():
    curated = {"A": {"joint_f1": 0.7486, "stability": 0.7049, "cls_f1": 0.7823}}
    live = {"A": {"joint_f1": 0.7000, "stability": 0.7049, "cls_f1": 0.7823}}
    rows = systems_view.merge(curated, live)
    a = next(r for r in rows if r["label"] == "A")
    assert a["joint_f1"] == 0.7486          # curated wins
    assert a["joint_f1_live"] == 0.7000
    assert a["mismatch"] is True            # differ beyond tolerance

def test_no_mismatch_when_equal():
    curated = {"A": {"joint_f1": 0.7486, "stability": 0.7049, "cls_f1": 0.7823}}
    live = {"A": {"joint_f1": 0.7486, "stability": 0.7049, "cls_f1": 0.7823}}
    rows = systems_view.merge(curated, live)
    assert rows[0]["mismatch"] is False

def test_g_annotated_not_comparable():
    curated = {"G": {"joint_f1": 0.4750, "stability": 0.4658, "cls_f1": None}}
    rows = systems_view.merge(curated, {})
    g = next(r for r in rows if r["label"] == "G")
    assert g["note"] == "span-only, not comparable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_systems_view.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write systems_view.py**

`agentic_os/backend/sources/systems_view.py`:
```python
from __future__ import annotations

TOL = 0.005  # 2-decimal reporting tolerance


def merge(curated: dict[str, dict], live: dict[str, dict]) -> list[dict]:
    """Curated metrics win; live merged for drill-down; flag mismatch > TOL."""
    rows = []
    for label in sorted(curated, key=lambda x: -(curated[x].get("joint_f1") or 0)):
        c = curated[label]
        lv = live.get(label, {})
        live_jf1 = lv.get("joint_f1")
        cur_jf1 = c.get("joint_f1")
        mismatch = (
            live_jf1 is not None and cur_jf1 is not None
            and abs(live_jf1 - cur_jf1) > TOL
        )
        rows.append({
            "label": label,
            "joint_f1": cur_jf1,
            "joint_f1_live": live_jf1,
            "stability": c.get("stability"),
            "cls_f1": c.get("cls_f1"),
            "mismatch": bool(mismatch),
            "note": "span-only, not comparable" if label == "G" else None,
        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_systems_view.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agentic_os/backend/sources/systems_view.py agentic_os/backend/tests/test_systems_view.py
git commit -m "feat(aos): merge curated+live system metrics with mismatch flag"
```

---

## Task 6: project_overview.py parser — experiments + blockers

**Files:**
- Create: `agentic_os/backend/parsers/project_overview.py`
- Test: `agentic_os/backend/tests/test_project_overview.py`
- Fixture: `agentic_os/backend/tests/fixtures/project_overview_sample.md`

- [ ] **Step 1: Create the fixture**

`agentic_os/backend/tests/fixtures/project_overview_sample.md` (real lines from memory):
```markdown
## Rigor Experiments (`Additional_Rigor_Experiments/`)

| # | Script | What | Status | Notes |
|---|--------|------|--------|-------|
| 01 | `run_01_bio_muril_recovery.sh` | BIO + MuRIL on Telugu | **SKIP** | Decoder fix. |
| 02 | `run_02_xlmr_qa_vs_bio.sh` | XLM-R Joint + BIO vs mBERT | ❌ Not run | Single biggest ceiling-mover. |
| 03 | `run_03_muril_joint.sh` | MuRIL Joint (System E) full training | ❌ Not run | After 02. |
| 04 | `run_04_llama3_baseline.py` | Llama-3.3-70B baseline | ❌ Not run | API only. |
| 05 | `run_05_decoder_rerun.py` | Decoder bug comparison table | ✅ Done | CPU only. |
| 06 | `run_06_idiombert_v2.py` | IdiomBERT-v2 8-cell ablation | ❌ Not run | Last. |
| 07 | TBD `run_07_bio_joint.py` | BIO Joint | ❌ Not planned yet | After 06. |

**Blocking TODOs:**
1. **IAA table (§5.1) — HARD BLOCKER.** Needs 2 annotators per language.
2. Error analysis table (§9) — user doing manual analysis.
3. Indonesian CI bug — Systems A–F return wrong key.
```

- [ ] **Step 2: Write the failing test**

`agentic_os/backend/tests/test_project_overview.py`:
```python
from pathlib import Path
from agentic_os.backend.parsers import project_overview as po

FIX = Path(__file__).parent / "fixtures" / "project_overview_sample.md"

def test_parse_experiments():
    exps = {e["id"]: e for e in po.parse_experiments(FIX.read_text())}
    assert exps["01"]["status"] == "skip"
    assert exps["02"]["status"] == "not_run"
    assert exps["05"]["status"] == "done"
    assert exps["02"]["script"] == "run_02_xlmr_qa_vs_bio.sh"

def test_parse_blockers_marks_hard():
    blockers = po.parse_blockers(FIX.read_text())
    iaa = next(b for b in blockers if "IAA" in b["title"])
    assert iaa["hard_blocker"] is True
    assert any(not b["hard_blocker"] for b in blockers)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_project_overview.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Write project_overview.py**

`agentic_os/backend/parsers/project_overview.py`:
```python
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
    """Parse the numbered 'Blocking TODOs' list."""
    out = []
    in_block = False
    for ln in text.splitlines():
        if "Blocking TODOs" in ln:
            in_block = True
            continue
        if in_block:
            m = re.match(r"\d+\.\s+(.*)", ln.strip())
            if not m:
                if ln.strip().startswith("##"):
                    break
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_project_overview.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add agentic_os/backend/parsers/project_overview.py agentic_os/backend/tests/test_project_overview.py agentic_os/backend/tests/fixtures/project_overview_sample.md
git commit -m "feat(aos): parse experiments + blockers from project_overview.md"
```

---

## Task 7: Next-Step engine — derive ordered backlog

**Files:**
- Create: `agentic_os/backend/tasks/__init__.py` (empty)
- Create: `agentic_os/backend/tasks/derive.py`
- Test: `agentic_os/backend/tests/test_derive.py`

- [ ] **Step 1: Write the failing test**

`agentic_os/backend/tests/test_derive.py`:
```python
from agentic_os.backend.tasks import derive

EXPS = [
    {"id": "01", "script": "run_01.sh", "what": "x", "status": "skip", "notes": ""},
    {"id": "02", "script": "run_02.sh", "what": "x", "status": "not_run", "notes": ""},
    {"id": "03", "script": "run_03.sh", "what": "x", "status": "not_run", "notes": ""},
    {"id": "04", "script": "run_04.py", "what": "x", "status": "not_run", "notes": ""},
    {"id": "05", "script": "run_05.py", "what": "x", "status": "done", "notes": ""},
    {"id": "06", "script": "run_06.py", "what": "x", "status": "not_run", "notes": ""},
    {"id": "07", "script": "run_07.py", "what": "x", "status": "not_run", "notes": ""},
]
BLOCKERS = [
    {"title": "IAA table", "hard_blocker": True},
    {"title": "Error analysis table", "hard_blocker": False},
]

def test_iaa_is_first():
    tasks = derive.build_backlog(EXPS, BLOCKERS)
    assert tasks[0]["title"].startswith("IAA")

def test_skip_and_done_excluded():
    tasks = derive.build_backlog(EXPS, BLOCKERS)
    ids = [t["id"] for t in tasks]
    assert "exp-01" not in ids   # skip
    assert "exp-05" not in ids   # done

def test_exp_run_order():
    tasks = derive.build_backlog(EXPS, BLOCKERS)
    exp_ids = [t["id"] for t in tasks if t["id"].startswith("exp-")]
    # 02 and 03 before 04 before 06-dryrun before 06 before 07
    assert exp_ids.index("exp-02") < exp_ids.index("exp-04")
    assert exp_ids.index("exp-03") < exp_ids.index("exp-04")
    assert exp_ids.index("exp-04") < exp_ids.index("exp-06-dryrun")
    assert exp_ids.index("exp-06-dryrun") < exp_ids.index("exp-06")
    assert exp_ids.index("exp-06") < exp_ids.index("exp-07")

def test_next_step_is_first_with_action():
    nxt = derive.next_step(EXPS, BLOCKERS)
    assert nxt["title"].startswith("IAA")
    assert "why" in nxt and "next_action" in nxt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_derive.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write derive.py**

`agentic_os/backend/tasks/derive.py`:
```python
from __future__ import annotations  # required: `dict | None` return annotation on 3.9

# Run-order priority for rigor experiments (lower = sooner).
EXP_ORDER = {"02": 10, "03": 11, "04": 20, "06-dryrun": 30, "06": 31, "07": 40}

DRYRUN_CMD = ("python Additional_Rigor_Experiments/run_06_idiombert_v2.py "
              "--output_dir /tmp/_dryrun --epochs 1 --batch_size 8 --langs English")


def _blocker_tasks(blockers: list[dict]) -> list[dict]:
    tasks = []
    for b in blockers:
        tasks.append({
            "id": "blocker-" + b["title"].split()[0].lower(),
            "title": b["title"],
            "source_file": "project_overview.md",
            "why": "Hard blocker for submission." if b["hard_blocker"]
                   else "Open paper TODO.",
            "next_action": "Resolve in paper / annotation tool.",
            "status": "not_run",
            "blocked_by": [],
            "_prio": 0 if b["hard_blocker"] else 50,
        })
    return tasks


def _exp_tasks(exps: list[dict]) -> list[dict]:
    tasks = []
    for e in exps:
        if e["status"] in ("skip", "done", "not_planned"):
            continue
        eid = e["id"]
        # Inject the 06 dry-run as its own pre-task.
        if eid == "06":
            tasks.append({
                "id": "exp-06-dryrun",
                "title": "Exp 06 dry-run (1 epoch, English)",
                "source_file": "project_overview.md",
                "why": "CLAUDE.md requires a dry-run before the real 06 run.",
                "next_action": DRYRUN_CMD,
                "status": "not_run",
                "blocked_by": ["exp-04"],
                "_prio": 30 + EXP_ORDER.get("06-dryrun", 99),
            })
        tasks.append({
            "id": f"exp-{eid}",
            "title": f"Exp {eid}: {e['what']}",
            "source_file": "project_overview.md",
            "why": e["notes"] or "Rigor experiment.",
            "next_action": f"Run {e['script']}",
            "status": e["status"],
            "blocked_by": [],
            "_prio": 30 + EXP_ORDER.get(eid, 99),
        })
    return tasks


def build_backlog(exps: list[dict], blockers: list[dict]) -> list[dict]:
    tasks = _blocker_tasks(blockers) + _exp_tasks(exps)
    tasks.sort(key=lambda t: t["_prio"])
    for t in tasks:
        t.pop("_prio", None)
    return tasks


def next_step(exps: list[dict], blockers: list[dict]) -> dict | None:
    backlog = build_backlog(exps, blockers)
    for t in backlog:
        if t["status"] != "done":
            return t
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_derive.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agentic_os/backend/tasks/ agentic_os/backend/tests/test_derive.py
git commit -m "feat(aos): derive ordered backlog + next-step from project state"
```

---

## Task 8: Write-back with backup + read-back verify

**Files:**
- Create: `agentic_os/backend/tasks/writeback.py`
- Test: `agentic_os/backend/tests/test_writeback.py`

- [ ] **Step 1: Write the failing test**

`agentic_os/backend/tests/test_writeback.py`:
```python
from pathlib import Path
import pytest
from agentic_os.backend.tasks import writeback

OVERVIEW = """| # | Script | What | Status | Notes |
|---|--------|------|--------|-------|
| 02 | `run_02.sh` | XLM-R | ❌ Not run | note |
"""

def test_mark_experiment_done(tmp_path):
    f = tmp_path / "po.md"
    f.write_text(OVERVIEW)
    writeback.mark_experiment_done(f, "02")
    txt = f.read_text()
    assert "✅ Done" in txt
    assert "❌ Not run" not in txt
    # backup was created
    assert list(tmp_path.glob("po.md.bak.*"))

def test_readback_verify_failure_restores(tmp_path, monkeypatch):
    f = tmp_path / "po.md"
    f.write_text(OVERVIEW)
    original = f.read_text()
    # Force the post-write parse to claim "not done" -> must restore.
    monkeypatch.setattr(writeback, "_is_experiment_done", lambda text, eid: False)
    with pytest.raises(RuntimeError, match="read-back"):
        writeback.mark_experiment_done(f, "02")
    assert f.read_text() == original   # restored

def test_dry_run_does_not_write(tmp_path):
    f = tmp_path / "po.md"
    f.write_text(OVERVIEW)
    writeback.mark_experiment_done(f, "02", dry_run=True)
    assert f.read_text() == OVERVIEW

def test_missing_id_raises(tmp_path):
    f = tmp_path / "po.md"
    f.write_text(OVERVIEW)
    with pytest.raises(ValueError, match="99"):
        writeback.mark_experiment_done(f, "99")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_writeback.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write writeback.py**

`agentic_os/backend/tasks/writeback.py`:
```python
from __future__ import annotations

import re
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_writeback.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agentic_os/backend/tasks/writeback.py agentic_os/backend/tests/test_writeback.py
git commit -m "feat(aos): write-back exp status with backup + read-back verify"
```

---

## Task 9: git_activity + folders sources

**Files:**
- Create: `agentic_os/backend/sources/git_activity.py`
- Create: `agentic_os/backend/sources/folders.py`
- Test: `agentic_os/backend/tests/test_sources_misc.py`

- [ ] **Step 1: Write the failing test**

`agentic_os/backend/tests/test_sources_misc.py`:
```python
from agentic_os.backend.sources import git_activity, folders
from agentic_os.backend import config

def test_recent_commits_shape():
    commits = git_activity.recent_commits(config.REPO_ROOT, n=3)
    assert len(commits) <= 3
    assert all("hash" in c and "subject" in c for c in commits)

def test_folder_registry():
    regs = folders.registry()
    by_id = {f["id"]: f for f in regs}
    assert by_id["research"]["status"] == "live"
    assert by_id["papers"]["status"] == "stub"
    assert {"agents", "oscore"} <= set(by_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_sources_misc.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the two sources**

`agentic_os/backend/sources/git_activity.py`:
```python
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
```

`agentic_os/backend/sources/folders.py`:
```python
def registry() -> list[dict]:
    return [
        {"id": "research", "name": "Research", "icon": "🔬", "status": "live"},
        {"id": "papers", "name": "Papers", "icon": "📄", "status": "stub"},
        {"id": "agents", "name": "Agents", "icon": "🤖", "status": "stub"},
        {"id": "oscore", "name": "OS Core", "icon": "⚙️", "status": "stub"},
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_sources_misc.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agentic_os/backend/sources/git_activity.py agentic_os/backend/sources/folders.py agentic_os/backend/tests/test_sources_misc.py
git commit -m "feat(aos): git activity feed + folder registry sources"
```

---

## Task 10: FastAPI app + routes (read-only)

**Files:**
- Create: `agentic_os/backend/routes/__init__.py` (empty)
- Create: `agentic_os/backend/routes/research.py`
- Create: `agentic_os/backend/routes/system.py`
- Create: `agentic_os/backend/main.py`
- Test: `agentic_os/backend/tests/test_api_readonly.py`

- [ ] **Step 1: Write the failing test**

`agentic_os/backend/tests/test_api_readonly.py`:
```python
from fastapi.testclient import TestClient
from agentic_os.backend.main import app

client = TestClient(app)

def test_folders_endpoint():
    r = client.get("/api/folders")
    assert r.status_code == 200
    assert any(f["id"] == "research" for f in r.json())

def test_systems_endpoint():
    r = client.get("/api/research/systems")
    assert r.status_code == 200
    rows = r.json()
    assert any(row["label"] == "D" for row in rows)

def test_ablation_endpoint():
    r = client.get("/api/research/ablation")
    assert r.status_code == 200
    assert any(row["combo"] == "en_es_hi_te" for row in r.json())

def test_experiments_endpoint():
    r = client.get("/api/research/experiments")
    assert r.status_code == 200
    assert any(e["id"] == "02" for e in r.json())

def test_blockers_endpoint():
    r = client.get("/api/research/blockers")
    assert r.status_code == 200
    assert any(b["hard_blocker"] for b in r.json())

def test_activity_endpoint():
    r = client.get("/api/activity")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_api_readonly.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write routes/research.py**

`agentic_os/backend/routes/research.py`:
```python
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
```

- [ ] **Step 4: Write routes/system.py**

`agentic_os/backend/routes/system.py`:
```python
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
```

- [ ] **Step 5: Write main.py**

`agentic_os/backend/main.py`:
```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_api_readonly.py -v`
Expected: 6 passed. (TestClient triggers startup → config.validate against the real repo, which exists.)

- [ ] **Step 7: Commit**

```bash
git add agentic_os/backend/routes/ agentic_os/backend/main.py agentic_os/backend/tests/test_api_readonly.py
git commit -m "feat(aos): FastAPI app + read-only research/system routes"
```

---

## Task 11: tasks routes (next/tasks/done)

**Files:**
- Create: `agentic_os/backend/routes/tasks.py`
- Modify: `agentic_os/backend/main.py` (register router)
- Test: `agentic_os/backend/tests/test_api_tasks.py`

- [ ] **Step 1: Write the failing test**

`agentic_os/backend/tests/test_api_tasks.py`:
```python
from fastapi.testclient import TestClient
from agentic_os.backend.main import app

client = TestClient(app)

def test_next_endpoint():
    r = client.get("/api/next")
    assert r.status_code == 200
    body = r.json()
    assert body is None or ("title" in body and "next_action" in body)

def test_tasks_endpoint():
    r = client.get("/api/tasks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_done_unknown_id_returns_error():
    r = client.post("/api/tasks/exp-99/done")
    assert r.status_code == 200
    assert r.json().get("error") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_api_tasks.py -v`
Expected: FAIL — route not registered (404 / module error).

- [ ] **Step 3: Write routes/tasks.py**

`agentic_os/backend/routes/tasks.py`:
```python
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
```

- [ ] **Step 4: Register the router in main.py**

In `agentic_os/backend/main.py`, add import and registration:
```python
from agentic_os.backend.routes import research, system, tasks
```
and after the other `include_router` lines:
```python
app.include_router(tasks.router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests/test_api_tasks.py -v`
Expected: 3 passed.

> NOTE: `test_done_unknown_id_returns_error` uses `exp-99` so the writeback raises ValueError (id not in real table) and is caught → error body. It does NOT mutate the real file because `_edit_line` raises before any write.

- [ ] **Step 6: Run the full backend suite**

Run: `.venv/bin/python -m pytest agentic_os/backend/tests -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add agentic_os/backend/routes/tasks.py agentic_os/backend/main.py agentic_os/backend/tests/test_api_tasks.py
git commit -m "feat(aos): next/tasks/done task routes wired to write-back"
```

---

## Task 12: Frontend scaffold (vite + React + TS)

**Files:**
- Create: `agentic_os/frontend/` (vite scaffold)
- Create: `agentic_os/frontend/src/api.ts`

- [ ] **Step 1: Scaffold vite**

Run from repo root:
```bash
cd agentic_os && npm create vite@latest frontend -- --template react-ts && cd frontend && npm install
```
Expected: `agentic_os/frontend` created with deps installed.

- [ ] **Step 2: Add vitest + testing-library**

Run: `cd agentic_os/frontend && npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom`

In `agentic_os/frontend/package.json` `"scripts"`, add: `"test": "vitest run"`.

In `agentic_os/frontend/vite.config.ts`, add inside `defineConfig({...})`:
```ts
  test: { environment: "jsdom", globals: true, setupFiles: "./src/setupTests.ts" },
```
Create `agentic_os/frontend/src/setupTests.ts`:
```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 3: Write api.ts**

`agentic_os/frontend/src/api.ts`:
```ts
const BASE = "http://localhost:8011";

export type SystemRow = {
  label: string; joint_f1: number | null; joint_f1_live: number | null;
  stability: number | null; cls_f1: number | null; mismatch: boolean; note: string | null;
};
export type AblationRow = { combo: string; span_f1: number; indo_f1: number; stability: number };
export type Experiment = { id: string; script: string; what: string; status: string; notes: string };
export type Blocker = { title: string; hard_blocker: boolean };
export type Commit = { hash: string; subject: string; when: string };
export type Folder = { id: string; name: string; icon: string; status: "live" | "stub" };
export type Task = {
  id: string; title: string; source_file: string; why: string;
  next_action: string; status: string; blocked_by: string[];
};

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  return r.json() as Promise<T>;
}

export const api = {
  folders: () => get<Folder[]>("/api/folders"),
  systems: () => get<SystemRow[]>("/api/research/systems"),
  ablation: () => get<AblationRow[]>("/api/research/ablation"),
  experiments: () => get<Experiment[]>("/api/research/experiments"),
  blockers: () => get<Blocker[]>("/api/research/blockers"),
  activity: () => get<Commit[]>("/api/activity"),
  tasks: () => get<Task[]>("/api/tasks"),
  next: () => get<Task | null>("/api/next"),
  markDone: (id: string) =>
    fetch(`${BASE}/api/tasks/${id}/done`, { method: "POST" }).then((r) => r.json()),
};
```

- [ ] **Step 4: Verify build**

Run: `cd agentic_os/frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add agentic_os/frontend
git commit -m "feat(aos): frontend scaffold (vite+react+ts) and typed api client"
```

> NOTE: ensure `agentic_os/frontend/node_modules` is git-ignored. The repo `.gitignore` should already ignore `node_modules`; if not, add `agentic_os/frontend/node_modules/` to `.gitignore` in this commit.

---

## Task 13: OS shell — Desktop, Window, Taskbar, IconGrid

**Files:**
- Create: `agentic_os/frontend/src/os/Window.tsx`
- Create: `agentic_os/frontend/src/os/IconGrid.tsx`
- Create: `agentic_os/frontend/src/os/Taskbar.tsx`
- Create: `agentic_os/frontend/src/os/Desktop.tsx`
- Modify: `agentic_os/frontend/src/App.tsx`
- Test: `agentic_os/frontend/src/os/Desktop.test.tsx`

- [ ] **Step 1: Write the failing test**

`agentic_os/frontend/src/os/Desktop.test.tsx`:
```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import Desktop from "./Desktop";
import { api } from "../api";

vi.spyOn(api, "folders").mockResolvedValue([
  { id: "research", name: "Research", icon: "🔬", status: "live" },
  { id: "papers", name: "Papers", icon: "📄", status: "stub" },
]);

test("renders folder icons and opens a window on click", async () => {
  render(<Desktop />);
  const icon = await screen.findByText("Research");
  fireEvent.click(icon);
  await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument());
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agentic_os/frontend && npx vitest run src/os/Desktop.test.tsx`
Expected: FAIL — Desktop not found.

- [ ] **Step 3: Write Window.tsx**

`agentic_os/frontend/src/os/Window.tsx`:
```tsx
import { ReactNode, useState } from "react";

export default function Window(props: {
  title: string; onClose: () => void; children: ReactNode;
}) {
  const [pos, setPos] = useState({ x: 80, y: 80 });
  const onDrag = (e: React.MouseEvent) => {
    const sx = e.clientX, sy = e.clientY, ox = pos.x, oy = pos.y;
    const move = (m: MouseEvent) =>
      setPos({ x: ox + m.clientX - sx, y: oy + m.clientY - sy });
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };
  return (
    <div role="dialog" aria-label={props.title}
      style={{
        position: "absolute", left: pos.x, top: pos.y, width: 520, minHeight: 240,
        background: "#15151c", border: "1px solid #333", borderRadius: 8,
        boxShadow: "0 12px 40px rgba(0,0,0,.5)", color: "#eee", resize: "both",
        overflow: "auto",
      }}>
      <div onMouseDown={onDrag}
        style={{ display: "flex", justifyContent: "space-between", padding: "6px 10px",
          background: "#22222c", cursor: "move", borderBottom: "1px solid #333" }}>
        <strong>{props.title}</strong>
        <button onClick={props.onClose} aria-label="close">✕</button>
      </div>
      <div style={{ padding: 12 }}>{props.children}</div>
    </div>
  );
}
```

- [ ] **Step 4: Write IconGrid.tsx**

`agentic_os/frontend/src/os/IconGrid.tsx`:
```tsx
import { Folder } from "../api";

export default function IconGrid(props: {
  folders: Folder[]; onOpen: (id: string) => void;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 24, padding: 32 }}>
      {props.folders.map((f) => (
        <button key={f.id} onClick={() => props.onOpen(f.id)}
          style={{ width: 96, height: 96, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 6,
            background: "transparent", border: "none", color: "#eee", cursor: "pointer" }}>
          <span style={{ fontSize: 40 }}>{f.icon}</span>
          <span>{f.name}</span>
          {f.status === "stub" && <small style={{ opacity: 0.5 }}>soon</small>}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Write Taskbar.tsx**

`agentic_os/frontend/src/os/Taskbar.tsx`:
```tsx
export default function Taskbar(props: {
  open: string[]; onFocus: (id: string) => void;
}) {
  return (
    <div style={{ position: "fixed", bottom: 0, left: 0, right: 0, height: 40,
      background: "#0d0d12", borderTop: "1px solid #333", display: "flex",
      alignItems: "center", gap: 8, padding: "0 12px" }}>
      <strong style={{ color: "#7af" }}>◈ Agentic OS</strong>
      {props.open.map((id) => (
        <button key={id} onClick={() => props.onFocus(id)}
          style={{ background: "#22222c", color: "#eee", border: "1px solid #333",
            borderRadius: 4, padding: "2px 10px" }}>{id}</button>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Write Desktop.tsx**

`agentic_os/frontend/src/os/Desktop.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, Folder } from "../api";
import IconGrid from "./IconGrid";
import Window from "./Window";
import Taskbar from "./Taskbar";
import NextStepBar from "./NextStepBar";
import ResearchRoom from "../folders/research/ResearchRoom";
import StubRoom from "../folders/StubRoom";

export default function Desktop() {
  const [folders, setFolders] = useState<Folder[]>([]);
  const [open, setOpen] = useState<string[]>([]);

  useEffect(() => { api.folders().then(setFolders); }, []);

  const openFolder = (id: string) =>
    setOpen((o) => (o.includes(id) ? o : [...o, id]));
  const close = (id: string) => setOpen((o) => o.filter((x) => x !== id));

  const roomFor = (id: string) => {
    const f = folders.find((x) => x.id === id);
    if (id === "research") return <ResearchRoom />;
    return <StubRoom name={f?.name ?? id} />;
  };

  return (
    <div style={{ minHeight: "100vh", background:
      "radial-gradient(circle at 30% 20%, #1a1a2e, #0a0a0f)" }}>
      <NextStepBar />
      <IconGrid folders={folders} onOpen={openFolder} />
      {open.map((id) => (
        <Window key={id} title={folders.find((f) => f.id === id)?.name ?? id}
          onClose={() => close(id)}>
          {roomFor(id)}
        </Window>
      ))}
      <Taskbar open={open} onFocus={openFolder} />
    </div>
  );
}
```

- [ ] **Step 7: Replace App.tsx**

`agentic_os/frontend/src/App.tsx`:
```tsx
import Desktop from "./os/Desktop";
export default function App() { return <Desktop />; }
```

- [ ] **Step 8: Run test to verify it passes**

The test needs `NextStepBar`, `ResearchRoom`, `StubRoom` to exist as imports. Create minimal placeholders now (they are fully implemented in Tasks 14–16); a one-line stub keeps the shell test green:

`agentic_os/frontend/src/os/NextStepBar.tsx`:
```tsx
export default function NextStepBar() { return <div data-testid="nextbar" />; }
```
`agentic_os/frontend/src/folders/StubRoom.tsx`:
```tsx
export default function StubRoom(props: { name: string }) {
  return (
    <div>
      <h3>{props.name}</h3>
      <p style={{ opacity: 0.6 }}>Coming soon. Planned panels for this room:</p>
      <ul style={{ opacity: 0.6 }}>
        <li>Status overview</li><li>Key actions</li><li>Linked sources</li>
      </ul>
    </div>
  );
}
```
`agentic_os/frontend/src/folders/research/ResearchRoom.tsx`:
```tsx
export default function ResearchRoom() { return <div data-testid="research-room" />; }
```

Run: `cd agentic_os/frontend && npx vitest run src/os/Desktop.test.tsx`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add agentic_os/frontend/src
git commit -m "feat(aos): OS shell — desktop, draggable window, taskbar, icon grid"
```

---

## Task 14: Research room panels 1–5

**Files:**
- Create: `agentic_os/frontend/src/folders/research/SystemsBoard.tsx`
- Create: `agentic_os/frontend/src/folders/research/AblationMatrix.tsx`
- Create: `agentic_os/frontend/src/folders/research/ExperimentQueue.tsx`
- Create: `agentic_os/frontend/src/folders/research/Blockers.tsx`
- Create: `agentic_os/frontend/src/folders/research/Activity.tsx`
- Modify: `agentic_os/frontend/src/folders/research/ResearchRoom.tsx`
- Test: `agentic_os/frontend/src/folders/research/SystemsBoard.test.tsx`

- [ ] **Step 1: Write the failing test**

`agentic_os/frontend/src/folders/research/SystemsBoard.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { vi } from "vitest";
import SystemsBoard from "./SystemsBoard";
import { api } from "../../api";

vi.spyOn(api, "systems").mockResolvedValue([
  { label: "D", joint_f1: 0.7515, joint_f1_live: 0.7515, stability: 0.7061, cls_f1: 0.7823, mismatch: false, note: null },
  { label: "G", joint_f1: 0.4750, joint_f1_live: 0.4750, stability: 0.4658, cls_f1: null, mismatch: false, note: "span-only, not comparable" },
]);

test("renders systems with 2-decimal metrics and G note", async () => {
  render(<SystemsBoard />);
  expect(await screen.findByText("0.75")).toBeInTheDocument();
  expect(screen.getByText(/span-only, not comparable/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agentic_os/frontend && npx vitest run src/folders/research/SystemsBoard.test.tsx`
Expected: FAIL — SystemsBoard not found.

- [ ] **Step 3: Write SystemsBoard.tsx**

`agentic_os/frontend/src/folders/research/SystemsBoard.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, SystemRow } from "../../api";

const f2 = (x: number | null) => (x == null ? "—" : x.toFixed(2));

export default function SystemsBoard() {
  const [rows, setRows] = useState<SystemRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.systems().then((r) => {
      if ((r as any).error) setErr((r as any).detail);
      else setRows(r);
    });
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!rows) return <div>Loading…</div>;
  return (
    <table style={{ width: "100%", fontSize: 13 }}>
      <thead><tr><th>Sys</th><th>Joint F1</th><th>Stability</th><th>Cls F1</th><th></th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label}>
            <td>{r.label}</td>
            <td>{f2(r.joint_f1)}{r.mismatch && <span title={`live ${f2(r.joint_f1_live)}`}> ⚠</span>}</td>
            <td>{f2(r.stability)}</td>
            <td>{f2(r.cls_f1)}</td>
            <td style={{ opacity: 0.6 }}>{r.note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Write the other four panels**

`agentic_os/frontend/src/folders/research/AblationMatrix.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, AblationRow } from "../../api";

const heat = (v: number) => `rgba(90,170,255,${Math.max(0, Math.min(1, v))})`;

export default function AblationMatrix() {
  const [rows, setRows] = useState<AblationRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.ablation().then((r) => (r as any).error ? setErr((r as any).detail) : setRows(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!rows) return <div>Loading…</div>;
  return (
    <table style={{ width: "100%", fontSize: 12 }}>
      <thead><tr><th>Combo</th><th>Span F1</th><th>Indo F1</th><th>Stability</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.combo}>
            <td>{r.combo}</td>
            <td style={{ background: heat(r.span_f1) }}>{r.span_f1.toFixed(2)}</td>
            <td style={{ background: heat(r.indo_f1) }}>{r.indo_f1.toFixed(2)}</td>
            <td>{r.stability.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

`agentic_os/frontend/src/folders/research/ExperimentQueue.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, Experiment } from "../../api";

const COLOR: Record<string, string> = {
  done: "#5c5", not_run: "#fa5", skip: "#888", not_planned: "#888", unknown: "#888",
};

export default function ExperimentQueue() {
  const [exps, setExps] = useState<Experiment[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.experiments().then((r) => (r as any).error ? setErr((r as any).detail) : setExps(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!exps) return <div>Loading…</div>;
  return (
    <ul style={{ listStyle: "none", padding: 0, fontSize: 13 }}>
      {exps.map((e) => (
        <li key={e.id} style={{ padding: "4px 0" }}>
          <span style={{ color: COLOR[e.status] ?? "#888" }}>●</span>{" "}
          <strong>{e.id}</strong> {e.what} <em style={{ opacity: 0.6 }}>({e.status})</em>
        </li>
      ))}
    </ul>
  );
}
```

`agentic_os/frontend/src/folders/research/Blockers.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, Blocker } from "../../api";

export default function Blockers() {
  const [bl, setBl] = useState<Blocker[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.blockers().then((r) => (r as any).error ? setErr((r as any).detail) : setBl(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!bl) return <div>Loading…</div>;
  return (
    <ul style={{ fontSize: 13 }}>
      {bl.map((b, i) => (
        <li key={i} style={{ color: b.hard_blocker ? "#f77" : "#eee" }}>
          {b.hard_blocker && <strong>[HARD] </strong>}{b.title}
        </li>
      ))}
    </ul>
  );
}
```

`agentic_os/frontend/src/folders/research/Activity.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, Commit } from "../../api";

export default function Activity() {
  const [cs, setCs] = useState<Commit[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.activity().then((r) => (r as any).error ? setErr((r as any).detail) : setCs(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!cs) return <div>Loading…</div>;
  return (
    <ul style={{ fontSize: 12, listStyle: "none", padding: 0 }}>
      {cs.map((c) => (
        <li key={c.hash}><code>{c.hash}</code> {c.subject} <span style={{ opacity: 0.5 }}>{c.when}</span></li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 5: Compose ResearchRoom.tsx**

Replace `agentic_os/frontend/src/folders/research/ResearchRoom.tsx`:
```tsx
import SystemsBoard from "./SystemsBoard";
import AblationMatrix from "./AblationMatrix";
import ExperimentQueue from "./ExperimentQueue";
import Blockers from "./Blockers";
import Activity from "./Activity";
import TasksPanel from "./TasksPanel";

const Card = (p: { title: string; children: React.ReactNode }) => (
  <section style={{ border: "1px solid #333", borderRadius: 6, padding: 10, marginBottom: 10 }}>
    <h4 style={{ margin: "0 0 8px" }}>{p.title}</h4>
    {p.children}
  </section>
);

export default function ResearchRoom() {
  return (
    <div data-testid="research-room">
      <Card title="Systems"><SystemsBoard /></Card>
      <Card title="Ablation Matrix"><AblationMatrix /></Card>
      <Card title="Experiment Queue"><ExperimentQueue /></Card>
      <Card title="Blockers"><Blockers /></Card>
      <Card title="Tasks"><TasksPanel /></Card>
      <Card title="Activity"><Activity /></Card>
    </div>
  );
}
```

> `TasksPanel` is created in Task 15. To keep this task's build green, create a one-line stub now: `agentic_os/frontend/src/folders/research/TasksPanel.tsx` → `export default function TasksPanel(){return <div data-testid="tasks-panel"/>;}` (fully implemented next task).

- [ ] **Step 6: Run test to verify it passes**

Run: `cd agentic_os/frontend && npx vitest run src/folders/research/SystemsBoard.test.tsx`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agentic_os/frontend/src/folders/research
git commit -m "feat(aos): research room panels — systems, ablation, queue, blockers, activity"
```

---

## Task 15: Tasks panel + Next-Step bar (wired to write-back)

**Files:**
- Modify: `agentic_os/frontend/src/folders/research/TasksPanel.tsx`
- Modify: `agentic_os/frontend/src/os/NextStepBar.tsx`
- Test: `agentic_os/frontend/src/folders/research/TasksPanel.test.tsx`

- [ ] **Step 1: Write the failing test**

`agentic_os/frontend/src/folders/research/TasksPanel.test.tsx`:
```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import TasksPanel from "./TasksPanel";
import { api } from "../../api";

const T = (id: string, title: string, status = "not_run") => ({
  id, title, source_file: "x", why: "y", next_action: "do z", status, blocked_by: [],
});

test("lists tasks and marks one done, then refetches", async () => {
  const spy = vi.spyOn(api, "tasks")
    .mockResolvedValueOnce([T("exp-02", "Exp 02"), T("exp-04", "Exp 04")])
    .mockResolvedValueOnce([T("exp-04", "Exp 04")]);
  vi.spyOn(api, "markDone").mockResolvedValue({ ok: true, next: T("exp-04", "Exp 04") });

  render(<TasksPanel />);
  const doneBtn = await screen.findAllByRole("button", { name: /done/i });
  fireEvent.click(doneBtn[0]);
  await waitFor(() => expect(api.markDone).toHaveBeenCalledWith("exp-02"));
  await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agentic_os/frontend && npx vitest run src/folders/research/TasksPanel.test.tsx`
Expected: FAIL — TasksPanel has no button.

- [ ] **Step 3: Write TasksPanel.tsx**

`agentic_os/frontend/src/folders/research/TasksPanel.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, Task } from "../../api";

export default function TasksPanel() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const load = () =>
    api.tasks().then((r) => (r as any).error ? setErr((r as any).detail) : setTasks(r));
  useEffect(() => { load(); }, []);

  const done = async (id: string) => {
    const res = await api.markDone(id);
    if (res.error) { setErr(res.detail); return; }
    await load();
  };

  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  return (
    <ol style={{ fontSize: 13, paddingLeft: 18 }}>
      {tasks.map((t) => (
        <li key={t.id} style={{ marginBottom: 6 }}>
          <strong>{t.title}</strong>
          <div style={{ opacity: 0.6 }}>{t.why}</div>
          <code style={{ fontSize: 11 }}>{t.next_action}</code>{" "}
          {t.id.startsWith("exp-") && (
            <button aria-label={`done ${t.id}`} onClick={() => done(t.id)}>✓ done</button>
          )}
        </li>
      ))}
    </ol>
  );
}
```

- [ ] **Step 4: Write NextStepBar.tsx**

`agentic_os/frontend/src/os/NextStepBar.tsx`:
```tsx
import { useEffect, useState } from "react";
import { api, Task } from "../api";

export default function NextStepBar() {
  const [next, setNext] = useState<Task | null>(null);
  useEffect(() => {
    api.next().then((r) => setNext(r && (r as any).error ? null : r));
  }, []);
  return (
    <div style={{ position: "sticky", top: 0, zIndex: 50, background: "#101820",
      borderBottom: "1px solid #2a3a4a", padding: "8px 16px", color: "#cfe" }}>
      {next ? (
        <span>▶ <strong>Next:</strong> {next.title} — <em style={{ opacity: 0.7 }}>{next.why}</em>{" "}
          <code style={{ fontSize: 11 }}>{next.next_action}</code></span>
      ) : (
        <span>✓ No pending next step.</span>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd agentic_os/frontend && npx vitest run src/folders/research/TasksPanel.test.tsx`
Expected: PASS.

- [ ] **Step 6: Run full frontend suite + build**

Run: `cd agentic_os/frontend && npm run test && npm run build`
Expected: all tests pass, build succeeds.

- [ ] **Step 7: Commit**

```bash
git add agentic_os/frontend/src/folders/research/TasksPanel.tsx agentic_os/frontend/src/os/NextStepBar.tsx agentic_os/frontend/src/folders/research/TasksPanel.test.tsx
git commit -m "feat(aos): tasks panel + next-step bar wired to write-back endpoint"
```

---

## Task 16: README + end-to-end preview verification

**Files:**
- Create: `agentic_os/README.md`

- [ ] **Step 1: Write README.md**

`agentic_os/README.md`:
```markdown
# Personal Agentic OS

Local desktop-metaphor dashboard over the IdiomBERT / MultiIdiom research.

## Run

Backend (from repo root):
    source .venv/bin/activate
    .venv/bin/pip install fastapi "uvicorn[standard]" pytest httpx
    .venv/bin/python -m uvicorn agentic_os.backend.main:app --reload --port 8011

Frontend:
    cd agentic_os/frontend
    npm install
    npm run dev   # http://localhost:5173

## Tests
    .venv/bin/python -m pytest agentic_os/backend/tests -v
    cd agentic_os/frontend && npm run test

## Folders
- Research — live (systems, ablation, experiment queue, blockers, tasks, activity)
- Papers / Agents / OS Core — stubs

## Next-Step engine
Derives an ordered backlog from `project_overview.md` (IAA hard-blocker first, then
rigor exps in run order 02/03 → 04 → 06-dryrun → 06 → 07). Marking an experiment done
performs a backed-up, read-back-verified edit of `project_overview.md`.
```

- [ ] **Step 2: Start backend (background)**

Run: `.venv/bin/python -m uvicorn agentic_os.backend.main:app --port 8011` (background)
Verify: `curl -s localhost:8011/api/folders` returns the 4 folders.

- [ ] **Step 3: Preview verification**

Start frontend dev server (`cd agentic_os/frontend && npm run dev`), then use the preview tools:
- `preview_start` on `http://localhost:5173`
- `preview_console_logs` → confirm no errors
- `preview_snapshot` → confirm Next-Step bar shows IAA, Research icon present
- `preview_click` the Research icon → `preview_snapshot` confirms 6 panels render with real numbers
- `preview_screenshot` → attach as proof

- [ ] **Step 4: Commit**

```bash
git add agentic_os/README.md
git commit -m "docs(aos): README + run/verify instructions"
```

---

## Self-Review Notes (resolved during authoring)

- **Spec coverage:** shell (T13), 6 research panels (T14–15), Next-Step bar (T15), derive engine (T7), write-back w/ backup + read-back (T8), stub folders (T13 StubRoom), error cards (every panel), config validation (T1), key_numbers-wins merge + mismatch (T5), ablation from key_numbers only (T3) — all mapped.
- **Type consistency:** `mark_experiment_done(path, eid, dry_run)`, `merge(curated, live)`, `next_step(exps, blockers)`, `api.markDone(id)` names match across backend↔frontend.
- **Write-back safety:** only `exp-*` tasks are writable (blockers/IAA are read-only in v1 — they live as prose, not a flippable token; flagged as a known v1 limit). `exp-NN-dryrun` maps to underlying `NN` via `.replace("-dryrun","")`.
- **Known v1 limits:** blocker completion not write-back-able (prose, not token); window positions not persisted; no live Colab polling. All per spec non-goals.
