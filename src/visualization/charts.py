import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch

from src.utils.config import ROOT

COLORS = ["#5DD9C1", "#6FA7FF", "#F4C974", "#B99AFF", "#F38EAD", "#78CAD2", "#DBDD81", "#D57D45"]
px.defaults.color_discrete_sequence = COLORS


def style(figure, title=None, height=390):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111A29",
        plot_bgcolor="#111A29",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#DDE5F0"),
        colorway=COLORS,
        margin=dict(l=40, r=25, t=60, b=40),
        height=height,
        title=title or figure.layout.title.text,
        legend_title_text="",
    )
    figure.update_xaxes(gridcolor="#243045")
    figure.update_yaxes(gridcolor="#243045")
    return figure


def comparison_chart(summary, metric, title, unit, kind="all"):
    data = summary[summary.kind == kind].copy()
    data["plus"] = data[f"{metric}_ci_high"] - data[metric]
    data["minus"] = data[metric] - data[f"{metric}_ci_low"]
    fig = px.bar(
        data,
        x="controller",
        y=metric,
        color="controller",
        error_y="plus",
        error_y_minus="minus",
        color_discrete_sequence=COLORS,
        labels={metric: unit, "controller": "Controller"},
    )
    fig.update_layout(showlegend=False)
    return style(fig, title)


def timeline_chart(trains):
    entered = trains[trains.actual_entry_time.notna()].copy()
    entered["occupied_minutes"] = entered.observed_exit_or_cutoff - entered.actual_entry_time
    fig = go.Figure()
    for direction, label, color in [(0, "A → B", COLORS[0]), (1, "B → A", COLORS[1])]:
        data = entered[entered.direction == direction]
        fig.add_trace(
            go.Bar(
                x=data.occupied_minutes,
                base=data.actual_entry_time,
                y=data.train_id,
                orientation="h",
                name=label,
                marker_color=color,
                hovertemplate="%{y}<br>Entry %{base} min<br>Duration %{x} min<extra></extra>",
            )
        )
    fig.update_layout(barmode="overlay", xaxis_title="Simulation minute", yaxis_title="Train")
    suffix = ""
    if len(entered) and "controller" in entered and "scenario_id" in entered:
        suffix = f" · {entered.controller.iloc[0]} · {entered.scenario_id.iloc[0]}"
    return style(fig, "Section occupation" + suffix, max(400, min(1000, len(entered) * 19)))


def export_charts(result_directory=None):
    result_directory = Path(result_directory) if result_directory else ROOT / "results"
    directory = result_directory / "charts"
    directory.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(result_directory / "comparison_summary.csv")
    frames = pd.read_csv(result_directory / "queue_frames.csv")
    trains = pd.read_csv(result_directory / "train_metrics.csv")
    first_id = frames.scenario_id.iloc[0]
    controllers = frames.controller.unique()
    chosen = next((name for name in controllers if name.startswith("PPO")), controllers[0])
    frame = frames[(frames.scenario_id == first_id) & (frames.controller == chosen)]
    train = trains[(trains.scenario_id == first_id) & (trains.controller == chosen)]
    saved = []

    def write(name, interactive, static):
        interactive.write_html(directory / f"{name}.html", include_plotlyjs="directory")
        static.tight_layout()
        static.savefig(directory / f"{name}.png", dpi=160, facecolor="white")
        plt.close(static)
        saved.append(name)

    all_rows = summary[summary.kind == "all"]
    for name, metric, title, unit in [
        (
            "throughput",
            "throughput",
            "Control-horizon throughput · 95% scenario bootstrap CI",
            "Trains / hour",
        ),
        (
            "delay",
            "mean_delay",
            "Completed-train mean waiting · 95% scenario bootstrap CI",
            "Minutes",
        ),
        (
            "utilization",
            "utilization",
            "Section utilization · excludes clearance",
            "Fraction of control horizon",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        error = [
            all_rows[metric] - all_rows[f"{metric}_ci_low"],
            all_rows[f"{metric}_ci_high"] - all_rows[metric],
        ]
        ax.bar(all_rows.controller, all_rows[metric], yerr=error, color=COLORS[: len(all_rows)])
        ax.set(ylabel=unit, title=title)
        ax.tick_params(axis="x", rotation=25)
        write(name, comparison_chart(summary, metric, title, unit), fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label in [("queue_a", "Queue A"), ("queue_b", "Queue B")]:
        ax.step(frame.time, frame[column], where="post", label=label)
    ax.set(
        title=f"Queue evolution · {first_id} · {chosen}", xlabel="Minute", ylabel="Waiting trains"
    )
    ax.legend()
    interactive = style(
        px.line(frame, x="time", y=["queue_a", "queue_b"], line_shape="hv"), ax.get_title()
    )
    write("queues", interactive, fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    entered = train[train.actual_entry_time.notna()]
    ax.barh(
        entered.train_id,
        entered.observed_exit_or_cutoff - entered.actual_entry_time,
        left=entered.actual_entry_time,
        color=[COLORS[d] for d in entered.direction],
    )
    ax.set(title=f"Executed section occupations · {first_id} · {chosen}", xlabel="Minute")
    ax.legend(
        handles=[Patch(color=COLORS[0], label="A → B"), Patch(color=COLORS[1], label="B → A")],
        loc="lower right",
    )
    ax.tick_params(axis="y", labelsize=6)
    write("timeline", timeline_chart(train), fig)
    kinds = summary[(summary.kind != "all") & (summary.controller != "PPO seed mean")]
    fig, ax = plt.subplots(figsize=(11, 5))
    kinds.pivot(index="kind", columns="controller", values="throughput").plot.bar(
        ax=ax, color=COLORS
    )
    ax.set(title="Throughput by scenario type", ylabel="Trains / hour", xlabel="")
    ax.tick_params(axis="x", rotation=15)
    write(
        "scenarios",
        style(
            px.bar(kinds, x="kind", y="throughput", color="controller", barmode="group"),
            "Throughput by scenario type",
        ),
        fig,
    )
    experiment_path = result_directory / "experiment_metadata.json"
    registry = (
        json.loads(experiment_path.read_text()).get("model_details", {})
        if experiment_path.exists()
        else {}
    )
    if registry:
        episodes, validations = [], []
        for detail in registry.values():
            seed = str(detail["training_seed"])
            run_dir = (ROOT / detail["checkpoint"].replace("\\", "/")).parent
            episode = pd.read_csv(run_dir / "episodes.csv")
            episode["reward_smoothed"] = episode.reward.rolling(20, min_periods=1).mean()
            episode["seed"] = seed
            episodes.append(episode)
            validation = pd.read_csv(run_dir / "validation.csv")
            validation["seed"] = seed
            validations.append(validation)
        for name, tables, metric, title, unit in [
            (
                "learning_reward",
                episodes,
                "reward_smoothed",
                "Training reward · trailing 20-episode mean",
                "Episode reward",
            ),
            (
                "validation_delay",
                validations,
                "mean_delay",
                "Fixed-validation waiting delay",
                "Minutes",
            ),
        ]:
            data = pd.concat(tables)
            fig, ax = plt.subplots(figsize=(10, 5))
            for seed, group in data.groupby("seed"):
                ax.plot(group.timesteps, group[metric], label=f"Seed {seed}")
            ax.set(title=title, xlabel="Training transitions", ylabel=unit)
            ax.legend()
            write(name, style(px.line(data, x="timesteps", y=metric, color="seed"), title), fig)
    (directory / "index.json").write_text(json.dumps(saved, indent=2))
    print(f"Generated {len(saved)} measured charts, each in HTML and PNG.")
    return saved
