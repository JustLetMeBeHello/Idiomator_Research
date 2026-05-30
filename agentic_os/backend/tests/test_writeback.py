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
    # backup is removed after successful write (no .bak files should accumulate)
    assert not list(tmp_path.glob("po.md.bak.*"))

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


def test_done_lands_in_status_column_realistic_row(tmp_path):
    # realistic 5-col row with backticks + special chars in script/notes
    src = (
        "| # | Script | What | Status | Notes |\n"
        "|---|--------|------|--------|-------|\n"
        "| 02 | `run_02_xlmr.sh` | XLM-R Joint + BIO | ❌ Not run | biggest mover |\n"
    )
    f = tmp_path / "po.md"
    f.write_text(src)
    writeback.mark_experiment_done(f, "02")
    row = [l for l in f.read_text().splitlines() if l.strip().startswith("| 02 ")][0]
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[3] == "✅ Done"        # Status column
    assert cells[4] == "biggest mover"  # Notes preserved untouched
    assert cells[1] == "`run_02_xlmr.sh`"  # Script preserved


def test_malformed_short_row_raises_not_miswrites(tmp_path):
    # a row that startswith "| 02 " but has too few columns must raise, not corrupt
    src = "| 02 | only three |\n"
    f = tmp_path / "po.md"
    f.write_text(src)
    import pytest
    with pytest.raises(ValueError):
        writeback.mark_experiment_done(f, "02")
    # file unchanged, no backup left behind
    assert f.read_text() == src
    assert not list(tmp_path.glob("po.md.bak.*"))


def test_backup_removed_after_successful_write(tmp_path):
    src = (
        "| # | Script | What | Status | Notes |\n"
        "|---|--------|------|--------|-------|\n"
        "| 02 | `run_02.sh` | XLM-R | ❌ Not run | note |\n"
    )
    f = tmp_path / "po.md"
    f.write_text(src)
    writeback.mark_experiment_done(f, "02")
    # success path must NOT leave a permanent backup next to the real file
    assert not list(tmp_path.glob("po.md.bak.*"))
