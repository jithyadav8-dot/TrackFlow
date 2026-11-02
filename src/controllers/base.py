from typing import Protocol


class Controller(Protocol):
    def action(self, section) -> int: ...


def heads(section):
    masks = section.masks()
    return [section.queues[d][0] for d in (0, 1) if masks[d + 1]]
