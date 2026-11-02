from src.controllers.base import heads


class LongestWaiting:
    def action(self, section):
        choices = heads(section)
        return (
            min(choices, key=lambda t: (-(section.time - t.ready_time), t.train_id)).direction + 1
            if choices
            else 0
        )
