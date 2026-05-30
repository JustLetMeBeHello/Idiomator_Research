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
