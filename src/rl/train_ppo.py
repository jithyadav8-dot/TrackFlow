import json
import time

import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.scenario_generator import load_split
from src.rl.callbacks import ResearchCallback
from src.simulator.environment import TrainTrafficEnv
from src.utils.config import ROOT, ensure_dirs, load_config
from src.utils.reproducibility import digest, metadata, save_json, seed_all


def train(seed=42, timesteps=None, smoke=False, reward_override=None, label=None):
    ensure_dirs()
    seed_all(seed)
    config = load_config("ppo")["rl"]
    requested = timesteps or (20000 if smoke else config["total_timesteps"])
    schedule = pd.read_csv(ROOT / "data/processed/train_schedule.csv")
    validation = load_split("validation")
    reward = load_config()["reward"]
    reward.update(reward_override or {})
    name = label or ("smoke_" if smoke else "") + f"seed_{seed}"
    directory = ROOT / "models" / name
    if (directory / "metadata.json").exists():
        raise FileExistsError(f"Completed run already exists at {directory}; choose a new --label")
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / name
    check_env(TrainTrafficEnv(schedule=schedule), warn=True)
    env = DummyVecEnv(
        [
            lambda: Monitor(TrainTrafficEnv(schedule=schedule, reward_config=reward))
            for _ in range(config["n_envs"])
        ]
    )
    env.seed(seed)
    keys = [
        "learning_rate",
        "gamma",
        "gae_lambda",
        "n_steps",
        "batch_size",
        "n_epochs",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ]
    model = MaskablePPO(
        config["policy"],
        env,
        seed=seed,
        device=config["device"],
        policy_kwargs={
            "net_arch": config["network"]["hidden_layers"],
            "activation_fn": torch.nn.Tanh,
        },
        verbose=0,
        **{key: config[key] for key in keys},
    )
    model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))
    callback = ResearchCallback(
        directory, validation, min(config["validation_interval"], requested)
    )
    started = time.monotonic()
    model.learn(total_timesteps=requested, callback=callback)
    model.save(directory / "last.zip")
    reloaded = MaskablePPO.load(directory / "best.zip", device="cpu")
    eval_env = Monitor(TrainTrafficEnv(validation[0], reward_config=reward))
    reward_mean, reward_std = evaluate_policy(
        reloaded, eval_env, n_eval_episodes=2, deterministic=True, use_masking=True
    )
    obs = eval_env.reset()[0]
    action, _ = reloaded.predict(
        obs, deterministic=True, action_masks=eval_env.unwrapped.action_masks()
    )
    if not eval_env.unwrapped.action_masks()[int(action)]:
        raise AssertionError("Reloaded model produced an invalid masked action")
    details = {
        **metadata(),
        "algorithm": "MaskablePPO",
        "network": config["network"],
        "training_seed": seed,
        "requested_timesteps": requested,
        "actual_timesteps": model.num_timesteps,
        "elapsed_seconds": time.monotonic() - started,
        "config": config,
        "reward": reward,
        "configuration_checksum": digest({"rl": config, "reward": reward}),
        "dataset_checksum": digest((ROOT / "data/processed/train_schedule.csv").read_bytes()),
        "validation_scenario_checksums": [digest(s) for s in validation],
        "validation_objective": load_config("evaluation")["selection_objective"],
        "best_validation": min(callback.validations, key=lambda r: r["selection_cost"]),
        "checkpoint": (directory / "best.zip").relative_to(ROOT).as_posix(),
        "checkpoint_sha256": digest((directory / "best.zip").read_bytes()),
        "reload_masked_evaluation_reward": float(reward_mean),
        "reload_reward_std": float(reward_std),
        "stage": "smoke" if smoke else "training",
        "device": config["device"],
    }
    save_json(directory / "metadata.json", details)
    if not smoke and reward_override is None:
        registry_path = ROOT / "models/registry.json"
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        registry[str(seed)] = details
        save_json(registry_path, registry)
    env.close()
    eval_env.close()
    print(json.dumps(details, indent=2), flush=True)
    return details
