class SafetyError(RuntimeError):
    pass


def legal_dispatch(section, direction):
    return (
        direction in (0, 1)
        and section.occupied is None
        and section.time >= section.clear_until
        and bool(section.queues[direction])
    )


def authorize(section, direction):
    """Independent gate; every caller must pass here regardless of RL masks."""
    if not legal_dispatch(section, direction):
        return False
    train = section.queues[direction][0]
    if train.status != "WAITING" or train.ready_time > section.time:
        raise SafetyError("Attempt to enter before readiness or from invalid lifecycle state")
    if section.last_exit is not None and section.time < section.last_exit + section.clearance:
        raise SafetyError("Clearance violation")
    return True


def assert_invariants(section):
    if len({t.train_id for t in section.trains}) != section.total:
        raise SafetyError("Duplicate or missing train identity")
    waiting = [t for queue in section.queues for t in queue]
    queued_ids = {t.train_id for t in waiting}
    if len(queued_ids) != len(waiting):
        raise SafetyError("Duplicate queue membership")
    counts = dict.fromkeys(["NOT_READY", "WAITING", "IN_SECTION", "COMPLETED"], 0)
    for train in section.trains:
        if train.status not in counts:
            raise SafetyError("Unknown lifecycle state")
        counts[train.status] += 1
        if train.status in ("NOT_READY", "WAITING") and (
            train.actual_entry_time is not None or train.actual_exit_time is not None
        ):
            raise SafetyError("Premature actual timestamps")
        if train.status == "NOT_READY" and train.ready_time <= section.time:
            raise SafetyError("Ready train was omitted from queues")
        if train.status == "IN_SECTION" and train.actual_exit_time is not None:
            raise SafetyError("In-section train already has an exit")
        if (train.status == "WAITING") != (train.train_id in queued_ids):
            raise SafetyError("Queue/lifecycle mismatch")
        if (
            train.status in ("WAITING", "IN_SECTION", "COMPLETED")
            and train.ready_time > section.time
        ):
            raise SafetyError("Premature readiness")
        if train.actual_entry_time is not None and train.actual_entry_time < train.ready_time:
            raise SafetyError("Premature entry")
        if train.status in ("IN_SECTION", "COMPLETED") and train.actual_entry_time is None:
            raise SafetyError("Missing entry")
        if train.status == "COMPLETED":
            if (
                train.actual_exit_time is None
                or train.actual_exit_time - train.actual_entry_time != train.travel_duration
            ):
                raise SafetyError("Incorrect exit timing")
    if counts["IN_SECTION"] > 1 or counts["IN_SECTION"] != int(section.occupied is not None):
        raise SafetyError("Occupancy violation")
    if section.occupied is not None and section.occupied.status != "IN_SECTION":
        raise SafetyError("Occupancy/lifecycle mismatch")
    if section.occupied is not None and not any(t is section.occupied for t in section.trains):
        raise SafetyError("Unknown occupying train")
    if sum(counts.values()) != section.total or len(section.trains) != section.total:
        raise SafetyError("Train conservation failure")
    if counts["COMPLETED"] != section.completed:
        raise SafetyError("Completion counter mismatch")
    for direction, queue in enumerate(section.queues):
        keys = [(t.ready_time, t.train_id) for t in queue]
        if keys != sorted(keys) or any(t.direction != direction for t in queue):
            raise SafetyError("Invalid FIFO or direction")
