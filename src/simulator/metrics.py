from dataclasses import asdict

import numpy as np


def capacity_bound(trains, horizon, clearance):
    """Optimistic finite-horizon upper bound using shortest durations, ignoring arrivals."""
    used, count = 0, 0
    for duration in sorted(t.travel_duration for t in trains):
        used += duration + (clearance if count else 0)
        if used > horizon:
            break
        count += 1
    return count


def train_metrics(section):
    rows = []
    for train in section.trains:
        row = asdict(train)
        row["waiting_delay"] = (
            None if train.actual_entry_time is None else train.actual_entry_time - train.ready_time
        )
        row["schedule_relative_delay"] = (
            None
            if train.actual_entry_time is None
            else train.actual_entry_time - train.scheduled_entry_time
        )
        row["censored_wait"] = (
            max(0, section.time - train.ready_time) if train.actual_entry_time is None else 0
        )
        row["observed_exit_or_cutoff"] = (
            train.actual_exit_time
            if train.actual_exit_time is not None
            else section.time
            if train.actual_entry_time is not None
            else None
        )
        rows.append(row)
    return rows


def calculate_metrics(section):
    completed = [t for t in section.trains if t.status == "COMPLETED"]
    waits = [t.actual_entry_time - t.ready_time for t in completed]
    censored = [
        max(0, section.time - t.ready_time) for t in section.trains if t.actual_entry_time is None
    ]
    entered_waits = [
        t.actual_entry_time - t.ready_time
        for t in section.trains
        if t.actual_entry_time is not None
    ]
    bound = capacity_bound(section.trains, section.horizon, section.clearance)
    if section.completed_control > bound:
        raise AssertionError("Finite-horizon service capacity bound exceeded")
    incomplete = section.total - section.completed
    return {
        "total_trains": section.total,
        "completed_control": section.completed_control,
        "throughput": section.completed_control * 60 / section.horizon,
        "mean_delay": float(np.mean(waits)) if waits else None,
        "p95_delay": float(np.percentile(waits, 95)) if waits else None,
        "total_waiting_delay": sum(waits),
        "mean_schedule_relative_delay": float(
            np.mean([t.actual_entry_time - t.scheduled_entry_time for t in completed])
        )
        if completed
        else None,
        "utilization": section.busy_minutes / section.horizon,
        "clearance_blocked_minutes": section.clearance_minutes,
        "max_queue": section.max_queue,
        "average_queue": sum(section.queue_sums) / section.horizon,
        "average_queue_a": section.queue_sums[0] / section.horizon,
        "average_queue_b": section.queue_sums[1] / section.horizon,
        "max_wait": max(entered_waits + censored, default=0),
        "backlog_control": section.total - section.completed_control,
        "drain_completions": section.completed - section.completed_control,
        "incomplete_trains": incomplete,
        "censored_wait_total": sum(censored),
        "censored_train_count": len(censored),
        "accrued_wait_all_trains": section.accrued_wait,
        "invalid_actions": section.invalid_actions,
        "conflicting_attempts": section.conflicting_attempts,
        "occupancy_violations": 0,
        "conservation_violations": 0,
        "safety_violations": 0,
        "avoidable_holds": section.avoidable_holds,
        "hold_actions": section.action_counts[0],
        "dispatch_a_actions": section.action_counts[1],
        "dispatch_b_actions": section.action_counts[2],
        "capacity_bound_completions": bound,
        "final_time": section.time,
        "selection_cost": (section.accrued_wait + section.cutoff * incomplete)
        / max(section.total, 1),
    }
