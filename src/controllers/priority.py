from src.controllers.base import heads


class FixedPriority:
    def action(self, section):
        choices = heads(section)
        return (
            min(choices, key=lambda t: (-t.priority, t.ready_time, t.train_id)).direction + 1
            if choices
            else 0
        )
