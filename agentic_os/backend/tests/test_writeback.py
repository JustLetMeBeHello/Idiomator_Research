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
