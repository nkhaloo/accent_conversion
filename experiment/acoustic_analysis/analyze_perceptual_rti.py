#!/usr/bin/env python3
"""Compare output-to-timbre RTI using the perceptual-analysis facets.

The unit of analysis is an output--timbre pair. Segment/feature RTIs are first
collapsed to a median for each output, preventing recordings with more valid
VoiceSauce measurements from receiving more weight. Lower RTI means that the
output is acoustically closer to its timbre reference.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_perceptual_correlation import load_gender_maps, reference_normalizers
from compute_rti import parse


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "rti_full.csv"
DEFAULT_SUMMARY = HERE / "perceptual_rti_summary.csv"
DEFAULT_FIGURE = HERE.parent / "figures" / "perceptual_rti_output_timbre.png"
DEFAULT_OVERALL_SUMMARY = HERE / "perceptual_rti_overall_summary.csv"
DEFAULT_OVERALL_FIGURE = HERE.parent / "figures" / "perceptual_rti_overall_output_timbre.png"
RAW_INPUT = HERE / "output.csv"

MODEL_LABELS = {"openvoice": "OpenVoice", "seed_vc": "Seed-VC"}
GROUP_LABELS = {"top5": "Top 5", "bottom5": "Bottom 5"}
GROUP_COLORS = {"top5": "#3F6CC1", "bottom5": "#D76837"}
FEATURES = [
    "H1H2c", "H2H4c", "H42Kc", "H2KH5Kc",
    "HNR05", "HNR15", "HNR25", "HNR35",
    "sF1", "sF2", "sF3", "sF4", "sB1", "sB2", "sB3", "sB4", "sF0",
    "CPP", "Energy",
]
PANELS = [
    ("openvoice", "top5"),
    ("seed_vc", "top5"),
    ("openvoice", "bottom5"),
    ("seed_vc", "bottom5"),
]


def rti(output_mean, reference_mean, reference_sd, denominator_floor=None):
    """Return non-negative Relative Transfer Index values.

    RTI = abs(output_mean - reference_mean) / reference_sd. If a floor is
    supplied, the denominator is max(reference_sd, floor). Inputs may be
    scalars, NumPy arrays, or pandas Series; invalid/zero denominators yield
    NaN rather than infinity.
    """
    output = np.asarray(output_mean, dtype=float)
    reference = np.asarray(reference_mean, dtype=float)
    denominator = np.asarray(reference_sd, dtype=float)
    if denominator_floor is not None:
        denominator = np.maximum(denominator, np.asarray(denominator_floor, dtype=float))
    valid = np.isfinite(output) & np.isfinite(reference) & np.isfinite(denominator) & (denominator > 0)
    result = np.full(np.broadcast(output, reference, denominator).shape, np.nan, dtype=float)
    np.divide(np.abs(output - reference), denominator, out=result, where=valid)
    return result.item() if result.ndim == 0 else result


def load_output_timbre_rti(path=DEFAULT_INPUT, raw_path=RAW_INPUT):
    """Recompute RTI after reference-derived phone × gender z-scoring."""
    data = pd.read_csv(path, low_memory=False)
    required = {
        "model", "rank", "pair_id", "feature", "reference_type", "O_mean", "R_mean",
        "segment", "reference_id", "R_sd",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data["model"] = data["model"].astype(str).str.strip().str.lower()
    data["rank"] = data["rank"].astype(str).str.strip().str.lower()
    data["reference_type"] = data["reference_type"].astype(str).str.strip().str.lower()
    data = data.loc[
        data["model"].isin(MODEL_LABELS)
        & data["rank"].isin(GROUP_LABELS)
        & data["reference_type"].eq("timbre")
    ].copy()
    raw = pd.read_csv(raw_path, low_memory=False)
    metadata = {filename: parse(filename) for filename in raw["Filename"].unique()}
    gender_maps = load_gender_maps()
    normalizers = reference_normalizers(raw, metadata, gender_maps)
    timbre_genders = gender_maps["timbre"]
    data["gender"] = data["reference_id"].map(timbre_genders)
    keys = pd.MultiIndex.from_frame(
        data[["reference_type", "segment", "gender", "feature"]],
        names=["reference_type", "phone", "gender", "feature"],
    )
    parameters = normalizers.reindex(keys).reset_index(drop=True)
    means = parameters["mean"].to_numpy()
    sds = parameters["sd"].to_numpy()
    data["O_mean_z"] = (data["O_mean"].to_numpy() - means) / sds
    data["R_mean_z"] = (data["R_mean"].to_numpy() - means) / sds
    data["R_sd_z"] = data["R_sd"].to_numpy() / sds
    floors = data.groupby("feature", observed=True)["R_sd_z"].transform(
        lambda values: values.quantile(0.05)
    )
    data["R_sd_z_floored"] = np.maximum(data["R_sd_z"], floors)
    data["RTI"] = rti(data["O_mean_z"], data["R_mean_z"], data["R_sd_z_floored"])
    data["normalization"] = "reference_phone_gender_zscore"
    return data.loc[data["RTI"].notna()]


def bootstrap_median_interval(values, rng, repetitions=5000):
    """Return a pair-level percentile bootstrap CI for the median."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    return tuple(np.quantile(np.median(draws, axis=1), [0.025, 0.975]))


def summarize_pairs(data, repetitions=5000, seed=20260806):
    """Collapse measurements to pair-feature values and summarize each facet."""
    pair_values = (
        data.groupby(["model", "rank", "pair_id", "feature"], observed=True)["RTI"]
        .median()
        .rename("pair_median_rti")
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows = []
    for model, rank in PANELS:
        for feature in FEATURES:
            values = pair_values.loc[
                pair_values["model"].eq(model)
                & pair_values["rank"].eq(rank)
                & pair_values["feature"].eq(feature),
                "pair_median_rti",
            ].to_numpy()
            if not len(values):
                raise ValueError(f"No output-to-timbre RTI values for {model}/{rank}/{feature}")
            low, high = bootstrap_median_interval(values, rng, repetitions)
            rows.append({
                "model": model,
                "rank": rank,
                "feature": feature,
                "median_rti": np.median(values),
                "ci_low": low,
                "ci_high": high,
                "output_timbre_pairs": len(values),
            })
    return pair_values, pd.DataFrame(rows)


def plot_facets(pair_values, summary, path=DEFAULT_FIGURE):
    """Draw per-feature median RTI bars in model × Top/Bottom-5 facets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    y_max = summary["ci_high"].max() * 1.18
    x = np.arange(len(FEATURES))
    for ax, (model, rank) in zip(axes.flat, PANELS):
        rows = (
            summary.loc[summary["model"].eq(model) & summary["rank"].eq(rank)]
            .set_index("feature")
            .loc[FEATURES]
        )
        medians = rows["median_rti"].to_numpy()
        ax.bar(
            x,
            medians,
            width=0.72,
            yerr=np.vstack([medians - rows["ci_low"], rows["ci_high"] - medians]),
            color=GROUP_COLORS[rank],
            alpha=0.92,
            capsize=2,
            error_kw={"elinewidth": 1.0, "ecolor": "#242933"},
            zorder=2,
        )
        ax.set_title(f"{MODEL_LABELS[model]} — {GROUP_LABELS[rank]}", loc="left", fontweight="bold")
        ax.set_xlim(-0.7, len(FEATURES) - 0.3)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x, FEATURES, rotation=45, ha="right", fontsize=8)
        ax.yaxis.grid(True, color="#D9DDE7", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_ylabel("Median output–timbre RTI (lower is closer)")
    axes[1, 0].set_ylabel("Median output–timbre RTI (lower is closer)")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10, hspace=0.42, wspace=0.16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_overall(pair_values, repetitions=5000, seed=20260806):
    """Collapse feature medians to one RTI value per output pair and facet."""
    pairs = (
        pair_values.groupby(["model", "rank", "pair_id"], observed=True)["pair_median_rti"]
        .median().reset_index(name="pair_median_rti")
    )
    rng = np.random.default_rng(seed)
    rows = []
    for model, rank in PANELS:
        values = pairs.loc[pairs["model"].eq(model) & pairs["rank"].eq(rank), "pair_median_rti"]
        low, high = bootstrap_median_interval(values, rng, repetitions)
        rows.append({
            "model": model, "rank": rank, "normalization": "reference_phone_gender_zscore",
            "median_rti": values.median(), "ci_low": low, "ci_high": high,
            "output_timbre_pairs": len(values),
        })
    return pd.DataFrame(rows)


def plot_overall_facets(summary, path=DEFAULT_OVERALL_FIGURE):
    """Draw one overall median RTI bar in each model × tier facet."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    y_max = summary["ci_high"].max() * 1.2
    for ax, (model, rank) in zip(axes.flat, PANELS):
        row = summary.loc[summary["model"].eq(model) & summary["rank"].eq(rank)].iloc[0]
        median = row["median_rti"]
        bars = ax.bar(
            [0], [median], width=0.58,
            yerr=[[median - row["ci_low"]], [row["ci_high"] - median]],
            color=GROUP_COLORS[rank], alpha=0.92, capsize=6,
            error_kw={"elinewidth": 1.6, "ecolor": "#242933"},
        )
        ax.bar_label(bars, labels=[f"{median:.2f}"], padding=8, fontweight="bold")
        ax.set_title(f"{MODEL_LABELS[model]} — {GROUP_LABELS[rank]}", loc="left", fontweight="bold")
        ax.set_xticks([0], ["Output–Timbre"])
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(0, y_max)
        ax.yaxis.grid(True, color="#D9DDE7", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_ylabel("Pair-level median RTI (lower is closer)")
    axes[1, 0].set_ylabel("Pair-level median RTI (lower is closer)")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.1, hspace=0.42, wspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_output_timbre_rti()
    pair_values, summary = summarize_pairs(data)
    overall_summary = summarize_overall(pair_values)
    DEFAULT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEFAULT_SUMMARY, index=False)
    overall_summary.to_csv(DEFAULT_OVERALL_SUMMARY, index=False)
    plot_facets(pair_values, summary)
    plot_overall_facets(overall_summary)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved summary to {DEFAULT_SUMMARY}")
    print(f"Saved figure to {DEFAULT_FIGURE}")
    print("\nOverall median RTI:")
    print(overall_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved overall summary to {DEFAULT_OVERALL_SUMMARY}")
    print(f"Saved overall figure to {DEFAULT_OVERALL_FIGURE}")


if __name__ == "__main__":
    main()
