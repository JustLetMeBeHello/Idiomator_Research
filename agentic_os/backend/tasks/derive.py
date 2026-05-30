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
