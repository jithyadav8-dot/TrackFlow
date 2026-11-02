from dataclasses import dataclass

STATES = ("NOT_READY", "WAITING", "IN_SECTION", "COMPLETED")


@dataclass
class Train:
    train_id: str
    direction: int
    scheduled_entry_time: int
    ready_time: int
    travel_duration: int
    priority: int = 1
    actual_entry_time: int | None = None
    actual_exit_time: int | None = None
    status: str = "NOT_READY"

    def __post_init__(self):
        for field in [
            "direction",
            "scheduled_entry_time",
            "ready_time",
            "travel_duration",
            "priority",
        ]:
            if not isinstance(getattr(self, field), int):
                raise ValueError(f"{field} must be an integer")
        if self.direction not in (0, 1) or self.travel_duration < 1 or self.priority < 1:
            raise ValueError("Invalid train direction, duration or priority")
        if self.ready_time < self.scheduled_entry_time or self.scheduled_entry_time < 0:
            raise ValueError("Invalid train timing")
        if (
            self.status != "NOT_READY"
            or self.actual_entry_time is not None
            or self.actual_exit_time is not None
        ):
            raise ValueError("New trains must start NOT_READY without actual times")

    def transition(self, status):
        if self.status == "COMPLETED" or STATES.index(status) != STATES.index(self.status) + 1:
            raise ValueError(f"Illegal transition {self.status} -> {status}")
        self.status = status

    @classmethod
    def from_record(cls, row):
        return cls(
            **{
                key: row[key]
                for key in [
                    "train_id",
                    "direction",
                    "scheduled_entry_time",
                    "ready_time",
                    "travel_duration",
                    "priority",
                ]
            }
        )
