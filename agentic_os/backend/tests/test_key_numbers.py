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
