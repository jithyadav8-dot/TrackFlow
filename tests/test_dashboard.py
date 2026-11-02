from streamlit.testing.v1 import AppTest

from src.utils.config import ROOT
from src.visualization.dashboard_components import advance_playback, initialize_playback


def test_playback_bounds():
    state = {}
    initialize_playback(state)
    state.update(frames=[{}, {}, {}], playing=True)
    advance_playback(state, 1)
    assert state["frame_index"] == 1 and state["playing"]
    state["playing"] = False
    advance_playback(state, 1)
    assert state["frame_index"] == 1
    state["playing"] = True
    advance_playback(state, 10)
    assert state["frame_index"] == 2 and not state["playing"]


def test_all_pages_render():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in [
        "Dataset Explorer",
        "Methodology",
        "Training Analytics",
        "RL vs Baselines",
        "Live Simulation",
    ]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Page {page} failed: {app.exception}"


def test_run_pause_resume_reset():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Live Simulation").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.session_state["frames"]) > 1
    app.button[1].click().run()
    assert app.session_state["playing"] is False
    position = app.session_state["frame_index"]
    app.button[2].click().run()
    assert app.session_state["playing"] is True
    assert app.session_state["frame_index"] > position
    app.button[3].click().run()
    assert not app.exception
    assert app.session_state["frames"] == []
    assert app.session_state["frame_index"] == 0
