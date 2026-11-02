from src.controllers.base import heads


class FCFS:
    def action(self, section):
        choices = heads(section)
        return (
            min(choices, key=lambda t: (t.ready_time, t.train_id)).direction + 1 if choices else 0
        )
