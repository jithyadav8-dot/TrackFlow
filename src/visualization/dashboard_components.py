import json

import streamlit as st

from src.utils.config import ROOT


def read_json(relative, default=None):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def model_registry():
    registry = read_json("models/registry.json", {})
    return {
        seed: info
        for seed, info in registry.items()
        if (ROOT / info["checkpoint"].replace("\\", "/")).is_file()
    }


def metric_row(values):
    for column, (title, value, help_text) in zip(st.columns(len(values)), values):
        column.metric(title, value, help=help_text)


def initialize_playback(state):
    for key, value in {
        "playing": False,
        "frame_index": 0,
        "frames": [],
        "live_metrics": None,
        "live_label": "",
        "scrub": 0,
    }.items():
        if key not in state:
            state[key] = value


def advance_playback(state, speed):
    if state["playing"] and state["frames"]:
        state["frame_index"] = min(len(state["frames"]) - 1, state["frame_index"] + speed)
        if state["frame_index"] == len(state["frames"]) - 1:
            state["playing"] = False
