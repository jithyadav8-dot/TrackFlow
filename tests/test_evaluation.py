import numpy as np
import pytest

from src.controllers import FCFS
from src.evaluation.experiments import run_episode
from src.evaluation.statistics import bootstrap_mean, percentage_improvement
from src.rl.policy_adapter import PPOController
from src.utils.config import ROOT
from tests.test_simulator import scenario


def test_paired_inputs_immutable_and_repeatable():
    from src.utils.reproducibility import digest

    data = scenario()
    before = digest(data)
    first = run_episode(data, FCFS(), record=True)
    assert first == run_episode(data, FCFS(), record=True)
    assert before == digest(data)
    assert len(first[2]) == first[0]["final_time"] + 1


def test_percentage_and_bootstrap():
    assert percentage_improvement(1, 0) is None
    assert percentage_improvement(8, 10, False) == 20
    assert percentage_improvement(12, 10) == 20
    assert bootstrap_mean([1, 2, 3]) == bootstrap_mean([1, 2, 3])
    assert np.isfinite(bootstrap_mean([1])).all()
    assert bootstrap_mean([]) == (None, None, None)


def test_missing_model_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        PPOController(tmp_path / "nonexistent.zip")


def test_model_reload_and_legal_actions():
    path = ROOT / "models/smoke_seed_42/best.zip"
    if not path.exists():
        pytest.skip("Run the actual PPO smoke training to create the model")
    controller = PPOController(path)
    metrics, _, _, _ = run_episode(scenario(), controller)
    assert metrics["invalid_actions"] == 0
    assert metrics["safety_violations"] == 0
