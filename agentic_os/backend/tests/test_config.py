import pytest
from agentic_os.backend import config

def test_repo_root_exists():
    assert config.REPO_ROOT.is_dir()

def test_memory_dir_exists():
    assert config.MEMORY_DIR.is_dir()

def test_results_json_path_resolves():
    assert config.RESULTS_JSON.name == "pipeline_eval_results.json"

def test_validate_raises_on_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="MEMORY_DIR"):
        config.validate()
