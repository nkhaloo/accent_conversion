#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from compute_rti import FEATURES, base_phone, parse


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "output.csv"
DEFAULT_OUTPUTS = {
    "timbre": HERE / "perceptual_correlation_output_timbre_by_feature.csv",
    "style": HERE / "perceptual_correlation_output_style_by_feature.csv",
}
METADATA_DIR = HERE.parents[1] / "source_metadata"
MAD_K = 3.5
MODELS = ("openvoice", "seed_vc")
RANKS = ("top5", "bottom5")
REFERENCE_TYPES = ("timbre", "style")


def load_gender_maps(metadata_dir=METADATA_DIR):
    """Load speaker gender keyed by the IDs used in acoustic filenames."""
    maps = {}
    for reference_type, filename in (
        ("timbre", "accent_archive_metadata.csv"),
        ("style", "english_source_speakers.csv"),
    ):
        table = pd.read_csv(Path(metadata_dir) / filename)
        gender_source = table["sex"].fillna(table["age_sex"]) if "sex" in table else table["age_sex"]
        gender = gender_source.astype(str).str.lower().str.extract(r"(female|male)", expand=False)
        speaker_id = table["mp3_file"].map(lambda value: Path(str(value)).stem)
        maps[reference_type] = dict(zip(speaker_id, gender))
    return maps


def reference_normalizers(data, metadata, gender_maps):
    """Estimate reference-only means/SDs for each phone × gender × feature."""
    parts = []
    for filename, info in metadata.items():
        if not info or info["role"] not in ("reference", "source"):
            continue
        reference_type = "timbre" if info["role"] == "reference" else "style"
        gender = gender_maps[reference_type].get(info["speaker_id"])
        if pd.isna(gender) or gender is None:
            continue
        frames = data.loc[data["Filename"].eq(filename), ["Label", "t_ms", *FEATURES]].copy()
        frames["phone"] = frames["Label"].map(base_phone)
        frames["gender"] = gender
        frames["reference_type"] = reference_type

        frames["recording"] = f"{reference_type}:{info['sentence']}:{info['speaker_id']}"
        parts.append(frames)
    references = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["reference_type", "recording", "phone", "t_ms"]
    )
    long = references.melt(
        id_vars=["reference_type", "phone", "gender"],
        value_vars=FEATURES, var_name="feature", value_name="value",
    )
    return long.groupby(["reference_type", "phone", "gender", "feature"])["value"].agg(
        mean="mean", sd="std"
    )


def recording_tokens(data, filename):
    """Return ordered (base-phone, frame table) tokens for one recording."""
    frames = data.loc[data["Filename"].eq(filename)].sort_values("t_ms").copy()
    boundary = frames["Label"].ne(frames["Label"].shift()) | frames["t_ms"].diff().gt(2)
    frames["token"] = boundary.cumsum()
    return [
        (base_phone(token["Label"].iloc[0]), token)
        for _, token in frames.groupby("token", sort=False)
    ]


def align_token_sequences(output_tokens, reference_tokens):
    """Pair same-phone tokens in order using longest common subsequence."""
    n_output, n_reference = len(output_tokens), len(reference_tokens)
    lengths = np.zeros((n_output + 1, n_reference + 1), dtype=int)
    for i in range(n_output):
        for j in range(n_reference):
            if output_tokens[i][0] == reference_tokens[j][0]:
                lengths[i + 1, j + 1] = lengths[i, j] + 1
            else:
                lengths[i + 1, j + 1] = max(lengths[i, j + 1], lengths[i + 1, j])

    aligned = []
    i, j = n_output, n_reference
    while i and j:
        if output_tokens[i - 1][0] == reference_tokens[j - 1][0]:
            aligned.append((output_tokens[i - 1][1], reference_tokens[j - 1][1]))
            i -= 1
            j -= 1
        elif lengths[i - 1, j] >= lengths[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return aligned[::-1]


def mad_outlier_mask(values, k=MAD_K):
    """Return the vspy-style modified-z-score outlier mask."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros(len(values), dtype=bool)
    return np.abs(0.6745 * (values - median) / mad) > k


def pearson_metrics(output_values, reference_values):
    """Calculate Pearson r and R-squared after finite/MAD filtering."""
    output_values = np.asarray(output_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    finite = np.isfinite(output_values) & np.isfinite(reference_values)
    output_values, reference_values = output_values[finite], reference_values[finite]
    outliers = mad_outlier_mask(output_values) | mad_outlier_mask(reference_values)
    output_values, reference_values = output_values[~outliers], reference_values[~outliers]
    if len(output_values) < 3 or np.std(output_values) == 0 or np.std(reference_values) == 0:
        correlation = np.nan
    else:
        correlation = np.corrcoef(output_values, reference_values)[0, 1]
    return {
        "pearson_r": correlation,
        "r_squared": correlation**2 if np.isfinite(correlation) else np.nan,
        "n_valid_frames": len(output_values),
        "n_finite_frames": int(finite.sum()),
        "n_outliers_dropped": int(outliers.sum()),
    }


def collect_aligned_frames(data, gender_maps):
    """Create matched frame values for output--timbre and output--style pairs."""
    metadata = {filename: parse(filename) for filename in data["Filename"].unique()}
    normalizers = reference_normalizers(data, metadata, gender_maps)
    references = {}
    for filename, info in metadata.items():
        if info and info["role"] in ("reference", "source"):
            reference_type = "timbre" if info["role"] == "reference" else "style"
            key = (reference_type, info["sentence"], info["speaker_id"])
            references.setdefault(key, []).append(info)

    def pick_reference(candidates, model, rank):
        for candidate in candidates:
            if candidate["model"] == model and candidate["rank"] == rank:
                return candidate
        return candidates[0]

    rows, alignment_rows = [], []
    for output_file, info in metadata.items():
        if not info or info["role"] != "output":
            continue
        output_tokens = recording_tokens(data, output_file)
        for reference_type, speaker_field in (("timbre", "timbre_id"), ("style", "source_id")):
            candidates = references.get(
                (reference_type, info["sentence"], info[speaker_field]), []
            )
            if not candidates:
                continue
            reference_file = pick_reference(candidates, info["model"], info["rank"])["filename"]
            speaker_id = info[speaker_field]
            gender = gender_maps[reference_type].get(speaker_id)
            if pd.isna(gender) or gender is None:
                raise ValueError(f"Missing gender metadata for {reference_type} speaker {speaker_id}")
            reference_tokens = recording_tokens(data, reference_file)
            aligned_tokens = align_token_sequences(output_tokens, reference_tokens)
            alignment_rows.append({
                "model": info["model"], "rank": info["rank"],
                "reference_type": reference_type,
                "output_file": output_file, "reference_file": reference_file,
                "output_tokens": len(output_tokens), "reference_tokens": len(reference_tokens),
                "matched_tokens": len(aligned_tokens),
            })
            for output_token, reference_token in aligned_tokens:
                phone = base_phone(output_token["Label"].iloc[0])
                output_time = np.linspace(0, 1, len(output_token))
                reference_time = np.linspace(0, 1, len(reference_token))
                for feature in FEATURES:
                    try:
                        parameters = normalizers.loc[(reference_type, phone, gender, feature)]
                    except KeyError:
                        continue
                    if not np.isfinite(parameters["sd"]) or parameters["sd"] <= 0:
                        continue
                    output_values = output_token[feature].to_numpy(dtype=float)
                    reference_values = reference_token[feature].to_numpy(dtype=float)
                    valid_reference = np.isfinite(reference_values)
                    if valid_reference.sum() < 2:
                        continue
                    interpolated = np.interp(
                        output_time,
                        reference_time[valid_reference],
                        reference_values[valid_reference],
                    )
                    output_values = (output_values - parameters["mean"]) / parameters["sd"]
                    interpolated = (interpolated - parameters["mean"]) / parameters["sd"]
                    rows.extend(
                        (info["model"], info["rank"], reference_type, feature,
                         output_value, reference_value)
                        for output_value, reference_value in zip(output_values, interpolated)
                    )
    columns = [
        "model", "rank", "reference_type", "feature", "output_value", "reference_value"
    ]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(alignment_rows)


def summarize_correlations(aligned_frames):
    """Return one correlation row per model, tier, reference type, and feature."""
    rows = []
    for model in MODELS:
        for rank in RANKS:
            for reference_type in REFERENCE_TYPES:
                for feature in FEATURES:
                    subset = aligned_frames.loc[
                        aligned_frames["model"].eq(model)
                        & aligned_frames["rank"].eq(rank)
                        & aligned_frames["reference_type"].eq(reference_type)
                        & aligned_frames["feature"].eq(feature)
                    ]
                    metrics = pearson_metrics(subset["output_value"], subset["reference_value"])
                    rows.append({
                        "model": model, "rank": rank, "reference_type": reference_type,
                        "feature": feature,
                        "normalization": "reference_phone_gender_zscore",
                        **metrics,
                    })
    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(DEFAULT_INPUT, low_memory=False)
    gender_maps = load_gender_maps()
    aligned_frames, alignment = collect_aligned_frames(data, gender_maps)
    summary = summarize_correlations(aligned_frames)
    for reference_type, output_path in DEFAULT_OUTPUTS.items():
        reference_summary = summary.loc[
            summary["reference_type"].eq(reference_type)
        ].reset_index(drop=True)
        reference_summary.to_csv(output_path, index=False)
    alignment_summary = alignment.groupby(["model", "rank", "reference_type"])[
        ["output_tokens", "reference_tokens", "matched_tokens"]
    ].sum()
    print("Token alignment summary:")
    print(alignment_summary.to_string())
    print("\nPer-feature correlations:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSaved tables to:")
    for output_path in DEFAULT_OUTPUTS.values():
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
