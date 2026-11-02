from src.simulator.environment import TrainTrafficEnv
from src.simulator.metrics import calculate_metrics


def evaluate_scenarios(model, scenarios):
    """Cost-based selection with masks on every prediction, no test-set access."""
    results = []
    for scenario in scenarios:
        env = TrainTrafficEnv(scenario)
        observation, _ = env.reset()
        while True:
            action, _ = model.predict(
                observation, deterministic=True, action_masks=env.action_masks()
            )
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        results.append(calculate_metrics(env.section))
        env.close()
    return results
