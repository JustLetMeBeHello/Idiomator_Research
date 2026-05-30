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


MULTI = """## IdiomBERT

**Blocking TODOs:**
1. Error analysis table (§9) — manual.
2. HF checkpoint IDs.

## MultiIdiom

**Blocking TODOs:**
1. **IAA table (§5.1) — HARD BLOCKER.** Needs 2 annotators.
2. Annotator demographics paragraph.

## Next Section
"""

def test_parse_blockers_accumulates_all_sections():
    blockers = po.parse_blockers(MULTI)
    titles = [b["title"] for b in blockers]
    # captures BOTH sections (4 total), not just the first
    assert len(blockers) == 4
    assert any("IAA" in t for t in titles)
    iaa = next(b for b in blockers if "IAA" in b["title"])
    assert iaa["hard_blocker"] is True
    # a stray '## Next Section' with no list must not crash or add items
