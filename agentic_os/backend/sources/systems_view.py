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
