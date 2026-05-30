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
    assert exp_ids.index("exp-02") < exp_ids.index("exp-04")
    assert exp_ids.index("exp-03") < exp_ids.index("exp-04")
    assert exp_ids.index("exp-04") < exp_ids.index("exp-06-dryrun")
    assert exp_ids.index("exp-06-dryrun") < exp_ids.index("exp-06")
    assert exp_ids.index("exp-06") < exp_ids.index("exp-07")

def test_next_step_is_first_with_action():
    nxt = derive.next_step(EXPS, BLOCKERS)
    assert nxt["title"].startswith("IAA")
    assert "why" in nxt and "next_action" in nxt
