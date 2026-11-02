from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    time: int
    kind: str
    train_id: str
    direction: int
