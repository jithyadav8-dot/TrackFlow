import pandas as pd
import plotly.express as px
import streamlit as st

from src.controllers import CONTROLLERS
from src.data.scenario_generator import KINDS, generate_scenario
from src.evaluation.experiments import run_episode
from src.rl.policy_adapter import PPOController
from src.utils.config import ROOT
from src.visualization.charts import comparison_chart, style, timeline_chart
from src.visualization.dashboard_components import (
    advance_playback,
    initialize_playback,
    metric_row,
    model_registry,
    read_json,
)
from src.visualization.railway_view import railway_figure

st.set_page_config(page_title="TrackFlow · Rail Operations Lab", page_icon="🚆", layout="wide")
st.markdown(
    """<style>
.block-container{padding-top:4rem;max-width:1500px}
[data-testid="stMetric"]{background:#111A29;border:1px solid #253249;border-radius:12px;padding:18px}
[data-testid="stMetricLabel"]{color:#9BAEC5;font-size:.83rem}
h1{letter-spacing:-1.4px}h2{letter-spacing:-.6px}
[data-testid="stSidebar"]{border-right:1px solid #253249}
.eyebrow{color:#5DD9C1;font-size:.75rem;letter-spacing:3px;font-weight:600}
</style>""",
    unsafe_allow_html=True,
)
st.sidebar.markdown("## 🚆 TrackFlow")
st.sidebar.caption("RAIL OPERATIONS LAB")
page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Live Simulation",
        "RL vs Baselines",
        "Training Analytics",
        "Dataset Explorer",
        "Methodology",
    ],
    label_visibility="collapsed",
)
st.sidebar.divider()
registry = model_registry()
dataset = read_json("data/metadata/dataset.json", {})
st.sidebar.caption(f"{len(registry)} trained PPO seeds available")
st.sidebar.caption("Timetable-derived demand\n\nHypothetical single-track bottleneck")
st.sidebar.caption("Research simulator · not an operational railway control system")
st.markdown(
    '<div class="eyebrow">TRACKFLOW / OPERATIONS INTELLIGENCE</div>', unsafe_allow_html=True
)
summary_path = ROOT / "results/comparison_summary.csv"
summary = pd.read_csv(summary_path) if summary_path.exists() else None
schedule_path = ROOT / "data/processed/train_schedule.csv"
schedule = pd.read_csv(schedule_path) if schedule_path.exists() else None

if page == "Overview":
    st.title("Every minute. Every movement.")
    st.write("A reproducible laboratory for train dispatch, queueing, and reinforcement learning.")
    metric_row(
        [
            (
                "Movements",
                str(dataset.get("train_count", "—")),
                "Selected four-hour base timetable",
            ),
            ("Section capacity", "1 train", "Shared by both directions"),
            ("Control window", "240 min", "Plus a 120-minute drain"),
            ("Trained seeds", str(len(registry)), "Completed runs with available checkpoints"),
        ]
    )
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(railway_figure({"time": 0}), use_container_width=True)
        st.caption("Topology illustration. Live traffic appears on the simulation page.")
    with right:
        st.subheader("Timetable meets a simulated constraint")
        st.write(
            "FCFS and fixed rules choose a ready direction. MaskablePPO learns the same choice from queue and section state. A separate safety gate enforces every movement."
        )
        st.info(
            f"Source: {dataset.get('source_type', 'not prepared')} · pair: {' ↔ '.join(dataset.get('station_pair', []))}"
        )
        st.caption(
            "GTFS describes planned service. The shared single track, clearance and disturbances are experimental assumptions."
        )
    if summary is not None:
        st.plotly_chart(
            comparison_chart(
                summary,
                "throughput",
                "Measured throughput across held-out scenarios",
                "Trains / hour",
            ),
            use_container_width=True,
        )
        st.caption(
            "95% bootstrap intervals across scenarios. PPO seed mean averages trained seeds within each scenario before bootstrapping."
        )
    else:
        st.info("Generate results with: python scripts/evaluate.py")

elif page == "Live Simulation":
    st.title("Dispatch desk")
    st.caption(
        "Run a deterministic simulation, then inspect its recorded minute-by-minute playback."
    )
    initialize_playback(st.session_state)
    if schedule is None:
        st.info("Prepare input first: python scripts/prepare_data.py")
        st.stop()
    controls = st.columns(4)
    choices = list(CONTROLLERS) + (["PPO"] if registry else [])
    controller_name = controls[0].selectbox("Controller", choices)
    kind = controls[1].selectbox(
        "Scenario", KINDS, format_func=lambda s: s.replace("_", " ").title()
    )
    count = controls[2].number_input("Train count", 0, 200, len(schedule), 1)
    seed = controls[3].number_input("Random seed", 0, 1000000, 42, 1)
    left, right = st.columns(2)
    intensity = left.slider("Delay intensity", 0.0, 3.0, 1.0, 0.25)
    speed = right.slider("Playback speed · minutes per frame", 1, 20, 3)
    if not registry:
        st.info(
            "PPO is unavailable until training finishes. Run python scripts/train.py. Baseline playback is ready."
        )
    seed_choice = (
        st.selectbox("PPO training seed", list(registry)) if controller_name == "PPO" else None
    )
    buttons = st.columns(4)
    if buttons[0].button("Run Simulation", type="primary", use_container_width=True):
        scenario = generate_scenario(schedule, kind, int(seed), int(count), intensity)
        controller = (
            PPOController(ROOT / registry[seed_choice]["checkpoint"].replace("\\", "/"))
            if seed_choice
            else CONTROLLERS[controller_name]()
        )
        with st.spinner("Simulating and checking safety…"):
            metrics, trains, frames, _ = run_episode(scenario, controller, record=True)
        st.session_state.update(
            frames=frames,
            frame_index=0,
            playing=True,
            live_metrics=metrics,
            live_trains=trains,
            live_label=f"{controller_name} · {kind} · seed {seed}",
            scrub=0,
        )
    if buttons[1].button("Pause", use_container_width=True, disabled=not st.session_state.frames):
        st.session_state.playing = False
    if buttons[2].button("Resume", use_container_width=True, disabled=not st.session_state.frames):
        st.session_state.playing = st.session_state.frame_index < len(st.session_state.frames) - 1
    if buttons[3].button("Reset", use_container_width=True):
        st.session_state.update(frames=[], frame_index=0, playing=False, live_metrics=None, scrub=0)

    def seek():
        st.session_state.frame_index = st.session_state.scrub
        st.session_state.playing = False

    if len(st.session_state.frames) > 1:
        st.slider(
            "Seek to minute", 0, len(st.session_state.frames) - 1, key="scrub", on_change=seek
        )

    @st.fragment(run_every=0.3 if st.session_state.playing else None)
    def playback():
        frames = st.session_state.frames
        if not frames:
            st.plotly_chart(railway_figure({"time": 0}), use_container_width=True)
            st.caption("Choose a scenario and run the simulation.")
            return
        frame = frames[st.session_state.frame_index]
        st.caption(
            st.session_state.live_label
            + (" · playing" if st.session_state.playing else " · paused")
        )
        metric_row(
            [
                ("Simulation clock", f"{frame['time']} min", "Current replay frame"),
                (
                    "Completed",
                    str(frame["completed"]),
                    "Includes drain completions after minute 240",
                ),
                ("Waiting", str(frame["queue_a"] + frame["queue_b"]), "Both directional queues"),
                (
                    "Mean wait",
                    f"{frame['mean_entered_wait']:.1f} min",
                    "Only trains entered by this replay frame",
                ),
            ]
        )
        st.plotly_chart(railway_figure(frame), use_container_width=True)
        st.progress(st.session_state.frame_index / max(1, len(frames) - 1))
        data = pd.DataFrame(frames[: st.session_state.frame_index + 1])
        st.plotly_chart(
            style(
                px.line(data, x="time", y=["queue_a", "queue_b"], line_shape="hv"),
                "Queue evolution",
            ),
            use_container_width=True,
        )
        was_playing = st.session_state.playing
        advance_playback(st.session_state, speed)
        if was_playing and not st.session_state.playing:
            st.rerun()

    playback()
    if st.session_state.live_metrics:
        with st.expander("Full-run outcome · includes future replay frames"):
            st.json(st.session_state.live_metrics)
            if st.session_state.live_trains:
                st.plotly_chart(
                    timeline_chart(pd.DataFrame(st.session_state.live_trains)),
                    use_container_width=True,
                )

elif page == "RL vs Baselines":
    st.title("Measure the difference")
    st.caption(
        "Identical saved train schedules, disturbances and safety rules for every controller."
    )
    if summary is None:
        st.info("Run python scripts/evaluate.py to generate measured comparisons.")
    else:
        kind = st.selectbox("Scenario filter", ["all", *KINDS])
        for column, metric, title, unit in [
            (0, "throughput", "Throughput", "Trains / hour"),
            (1, "mean_delay", "Completed-train mean waiting", "Minutes"),
            (0, "utilization", "Physical utilization", "Fraction"),
            (1, "incomplete_trains", "Unfinished at final cutoff", "Trains"),
        ]:
            if column == 0:
                pair = st.columns(2)
            with pair[column]:
                st.plotly_chart(
                    comparison_chart(summary, metric, title, unit, kind), use_container_width=True
                )
        st.dataframe(
            summary[summary.kind == kind][
                [
                    "controller",
                    "scenario_count",
                    "throughput",
                    "mean_delay",
                    "p95_delay",
                    "max_queue",
                    "average_queue",
                    "incomplete_trains",
                    "safety_violations",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Completed-train waiting omits unresolved trains. Inspect backlog and censored waits before drawing conclusions. Intervals quantify scenario variation, not deployment uncertainty."
        )
        paired_path = ROOT / "results/paired_differences.csv"
        if paired_path.exists() and paired_path.stat().st_size > 2:
            differences = pd.read_csv(paired_path)
            subset = differences if kind == "all" else differences[differences.kind == kind]
            st.subheader("Paired improvement percentages")
            st.dataframe(
                subset.groupby(["controller", "baseline", "metric"])
                .improvement_percent.mean()
                .rename("Mean scenario-wise improvement %")
                .reset_index(),
                hide_index=True,
            )
            with st.expander("Paired differences and 95% intervals · all test scenarios"):
                st.dataframe(pd.read_csv(ROOT / "results/paired_summary.csv"), hide_index=True)

elif page == "Training Analytics":
    st.title("Inside the learning process")
    if not registry:
        st.info(
            "No completed training runs yet. Run python scripts/train.py; the dashboard never starts training."
        )
    else:
        selected = st.selectbox("Training seed", list(registry))
        detail = registry[selected]
        run_dir = (ROOT / detail["checkpoint"].replace("\\", "/")).parent
        metric_row(
            [
                ("Algorithm", "MaskablePPO", "Action masks plus independent safety gate"),
                (
                    "Actual transitions",
                    f"{detail['actual_timesteps']:,}",
                    "Rounded to complete rollouts",
                ),
                (
                    "Best validation cost",
                    f"{detail['best_validation']['selection_cost']:.2f}",
                    "Accrued wait + backlog objective",
                ),
                ("Training seed", selected, "Independent initialization and training scenarios"),
            ]
        )
        episodes = pd.read_csv(run_dir / "episodes.csv")
        episodes["smoothed_reward"] = episodes.reward.rolling(20, min_periods=1).mean()
        st.plotly_chart(
            style(
                px.line(episodes, x="timesteps", y="smoothed_reward"),
                "Episode reward · trailing 20-episode mean",
            ),
            use_container_width=True,
        )
        validations = pd.read_csv(run_dir / "validation.csv")
        st.plotly_chart(
            style(
                px.line(validations, x="timesteps", y=["mean_delay", "selection_cost"]),
                "Fixed validation set",
            ),
            use_container_width=True,
        )
        progress_path = ROOT / "logs" / run_dir.name / "progress.csv"
        if progress_path.exists():
            progress = pd.read_csv(progress_path)
            losses = [
                name
                for name in [
                    "train/policy_gradient_loss",
                    "train/value_loss",
                    "train/entropy_loss",
                    "train/approx_kl",
                ]
                if name in progress
            ]
            for loss in losses:
                st.plotly_chart(
                    style(px.line(progress, x="time/total_timesteps", y=loss), loss, 270),
                    use_container_width=True,
                )
        with st.expander("Configuration, checkpoint and provenance"):
            st.json(detail)

elif page == "Dataset Explorer":
    st.title("Know the input")
    if schedule is None:
        st.info("Run python scripts/prepare_data.py to prepare the timetable.")
    else:
        metric_row(
            [
                ("Movements", str(len(schedule)), "Selected timetable window"),
                (
                    "A → B",
                    str((schedule.direction == 0).sum()),
                    "Station IDs are shown in the data",
                ),
                (
                    "B → A",
                    str((schedule.direction == 1).sum()),
                    "Inferred from station pair traversal",
                ),
                (
                    "Service date",
                    str(dataset.get("service_date", "Synthetic")),
                    "Source service date",
                ),
            ]
        )
        st.info(
            "Observed fields: published GTFS timetable and IDs. Derived fields: station pair, rounded arrival offsets, nominal traversal. Assumed fields: dispatch priorities, bottleneck, clearance, random delays and travel variation."
        )
        st.plotly_chart(
            style(
                px.histogram(
                    schedule,
                    x="scheduled_entry_min",
                    color=schedule.direction.astype(str),
                    nbins=24,
                ),
                "Timetable demand by direction",
            ),
            use_container_width=True,
        )
        st.dataframe(schedule, hide_index=True, use_container_width=True)
        with st.expander("Source, checksums and corridor ranking"):
            st.json(dataset)
        with st.expander("Preprocessing quality report"):
            st.json(read_json("data/metadata/quality.json", {}))

else:
    st.title("Methods, assumptions, limits")
    for title, file in [
        ("System architecture", "architecture.md"),
        ("Simulation and safety", "simulator.md"),
        ("Reinforcement learning", "rl_methodology.md"),
        ("Evaluation protocol", "evaluation.md"),
        ("Limitations", "limitations.md"),
    ]:
        path = ROOT / "docs" / file
        if path.exists():
            with st.expander(title, expanded=title == "System architecture"):
                st.markdown(path.read_text(encoding="utf-8"))
