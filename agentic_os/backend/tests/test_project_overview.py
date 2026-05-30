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
