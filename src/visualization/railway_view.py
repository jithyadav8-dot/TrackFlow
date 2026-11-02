import plotly.graph_objects as go

from src.visualization.charts import style


def railway_figure(frame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 10], y=[0, 0], mode="lines", line=dict(color="#40516B", width=9), hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 10],
            y=[0, 0],
            mode="markers+text",
            text=["STATION A", "STATION B"],
            textposition="bottom center",
            marker=dict(size=30, color=["#5DD9C1", "#6FA7FF"]),
            hoverinfo="skip",
        )
    )
    if frame.get("occupied"):
        fraction = min(1, max(0, (frame["time"] - frame["entry"]) / frame["duration"]))
        position = fraction * 10 if frame["direction"] == 0 else (1 - fraction) * 10
        fig.add_trace(
            go.Scatter(
                x=[position],
                y=[0],
                mode="markers+text",
                text=["A → B" if frame["direction"] == 0 else "B → A"],
                textposition="top center",
                marker=dict(symbol="square", size=24, color="#F4C974"),
                name=str(frame["occupied"]),
            )
        )
    for position, length, color in [
        (0.5, frame.get("queue_a", 0), "#5DD9C1"),
        (9.5, frame.get("queue_b", 0), "#6FA7FF"),
    ]:
        fig.add_annotation(
            x=position,
            y=0.55,
            text=f"{length} waiting",
            showarrow=False,
            font=dict(size=16, color=color),
        )
    fig.update_xaxes(visible=False, range=[-1, 11])
    fig.update_yaxes(visible=False, range=[-0.6, 1])
    fig.update_layout(showlegend=False)
    return style(fig, "Shared section · capacity 1", 250)
