from agentic_os.backend.sources import git_activity, folders
from agentic_os.backend import config

def test_recent_commits_shape():
    commits = git_activity.recent_commits(config.REPO_ROOT, n=3)
    assert len(commits) <= 3
    assert all("hash" in c and "subject" in c for c in commits)

def test_folder_registry():
    regs = folders.registry()
    by_id = {f["id"]: f for f in regs}
    assert by_id["research"]["status"] == "live"
    assert by_id["papers"]["status"] == "stub"
    assert {"agents", "oscore"} <= set(by_id)
