from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_LABELS = {
    "output_vs_timbreRef": "Output–Timbre",
    "output_vs_sourceRef": "Output–Style",
    "timbreRef_vs_sourceRef": "Timbre–Style",
    "timbreRef_vs_timbreRef": "Timbre–Timbre (Same Speaker)",
    "sourceRef_vs_sourceRef": "Style–Style (Same Speaker)",
    "sourceRef_vs_sourceRef_cross": "Style–Style (Different Speakers)",
}

OUTPUT_CONDITIONS = ["output_vs_timbreRef", "output_vs_sourceRef"]
CONTROL_CONDITIONS = [
    "timbreRef_vs_sourceRef",
    "timbreRef_vs_timbreRef",
    "sourceRef_vs_sourceRef",
    "sourceRef_vs_sourceRef_cross",
]
MODEL_LABELS = {"openvoice": "OpenVoice", "seed_vc": "SeedVC"}
GROUP_LABELS = {"top5": "Top 5", "bottom5": "Bottom 5"}
GROUP_COLORS = {"top5": "#3F6CC1", "bottom5": "#D76837"}
AXIS_LABEL = "Responses judged as same speaker (%)"


def load_and_clean(path):
    data = pd.read_csv(path, dtype=str)
    data.columns = data.columns.str.strip()
    for column in ["condition", "response", "model", "group", "participant", "session_id"]:
        data[column] = data[column].str.strip()
    data["response"] = data["response"].str.lower()
    data["model"] = data["model"].str.lower()
    data["group"] = data["group"].str.lower()
    data = data.loc[
        data["condition"].isin(CONDITION_LABELS)
        & data["response"].isin(["yes", "no"])
        & data["model"].isin(MODEL_LABELS)
    ].copy()
    data = data.drop_duplicates()
    data["cluster"] = data["participant"].fillna(data["session_id"])
    data["same_speaker"] = data["response"].eq("yes").astype(float)
    return data


def clustered_interval(data, rng, repetitions=5000):
    cluster_totals = data.groupby("cluster", observed=True)["same_speaker"].agg(["sum", "count"])
    values = cluster_totals[["sum", "count"]].to_numpy()
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    sampled = values[draws].sum(axis=1)
    estimates = sampled[:, 0] / sampled[:, 1] * 100
    return np.quantile(estimates, [0.025, 0.975])


def make_row(subset, panel, condition, series, rng):
    low, high = clustered_interval(subset, rng)
    return {
        "panel": panel,
        "condition": condition,
        "label": CONDITION_LABELS[condition],
        "series": series,
        "percent": subset["same_speaker"].mean() * 100,
        "low": low,
        "high": high,
        "responses": len(subset),
        "participants": subset["cluster"].nunique(),
    }


def summarize(data):
    rng = np.random.default_rng(20260806)
    rows = []
    for model, model_label in MODEL_LABELS.items():
        for group, group_label in GROUP_LABELS.items():
            for condition in OUTPUT_CONDITIONS:
                subset = data.loc[
                    (data["model"] == model)
                    & (data["group"] == group)
                    & (data["condition"] == condition)
                ]
                rows.append(make_row(subset, f"{model}:{group}", condition, f"{model_label}, {group_label}", rng))
    for condition in CONTROL_CONDITIONS:
        subset = data.loc[data["condition"] == condition]
        rows.append(make_row(subset, "controls", condition, "Pooled", rng))
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color="#D9DDE7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_bars(ax, rows, color):
    positions = np.arange(len(rows))
    bars = ax.barh(
        positions,
        rows["percent"],
        xerr=np.vstack([rows["percent"] - rows["low"], rows["high"] - rows["percent"]]),
        color=color,
        alpha=0.92,
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": "#242933"},
    )
    ax.set_yticks(positions, rows["label"])
    ax.invert_yaxis()
    for bar, row in zip(bars, rows.to_dict("records")):
        inside = row["percent"] >= 82
        ax.text(
            row["percent"] - 1.5 if inside else min(row["high"] + 1.5, 98),
            bar.get_y() + bar.get_height() / 2,
            f"{row['percent']:.1f}%",
            va="center",
            ha="right" if inside else "left",
            color="white" if inside else "#242933",
            fontsize=9,
            fontweight="bold",
        )
    style_axis(ax)


def make_output_figure(summary, path):
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42)
    panels = [
        ("openvoice:top5", "OpenVoice — Top 5", "top5", grid[0, 0]),
        ("seed_vc:top5", "SeedVC — Top 5", "top5", grid[0, 1]),
        ("openvoice:bottom5", "OpenVoice — Bottom 5", "bottom5", grid[1, 0]),
        ("seed_vc:bottom5", "SeedVC — Bottom 5", "bottom5", grid[1, 1]),
    ]
    axes = []
    for panel, title, group, location in panels:
        ax = fig.add_subplot(location)
        rows = summary.loc[summary["panel"] == panel].set_index("condition").loc[OUTPUT_CONDITIONS].reset_index()
        draw_bars(ax, rows, GROUP_COLORS[group])
        ax.set_title(title, loc="left", fontweight="bold")
        axes.append(ax)
    for ax in axes[-2:]:
        ax.set_xlabel(AXIS_LABEL)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_control_figure(summary, path):
    fig, control_ax = plt.subplots(figsize=(11, 5.5))
    controls = summary.loc[summary["panel"] == "controls"].set_index("condition").loc[CONTROL_CONDITIONS].reset_index()
    draw_bars(control_ax, controls, "#596273")
    control_ax.set_xlabel(AXIS_LABEL)
    fig.subplots_adjust(left=0.33, right=0.97, top=0.9, bottom=0.14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    experiment_dir = Path(__file__).resolve().parents[1]
    input_path = Path(__file__).resolve().parent / "results" / "speaker_identity_results.csv"
    output_path = experiment_dir / "figures" / "speaker_identity_model_outputs.png"
    control_path = experiment_dir / "figures" / "speaker_identity_controls.png"
    data = load_and_clean(input_path)
    summary = summarize(data)
    if summary[["percent", "low", "high"]].isna().any().any():
        raise ValueError("One or more expected model-tier-condition combinations are missing valid responses")
    make_output_figure(summary, output_path)
    make_control_figure(summary, control_path)
    print(summary[["label", "series", "percent", "low", "high", "responses", "participants"]].to_string(index=False))
    print(f"Saved figure to {output_path}")
    print(f"Saved figure to {control_path}")


if __name__ == "__main__":
    main()
