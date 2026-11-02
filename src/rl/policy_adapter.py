from pathlib import Path


class PPOController:
    def __init__(self, checkpoint):
        from sb3_contrib import MaskablePPO

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint}")
        self.model = MaskablePPO.load(str(checkpoint), device="cpu")

    def action(self, env):
        action, _ = self.model.predict(
            env.observation(), deterministic=True, action_masks=env.action_masks()
        )
        return int(action)
