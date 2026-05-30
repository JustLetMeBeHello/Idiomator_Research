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
