#!/usr/bin/env python3
"""Compute Relative Transfer Index (RTI) from VoiceSauce frame-level output.

RTI_{f,c,t} = (O_mean[f,c] - R_mean[f,c]) / R_sd[f,c]

where O is the model output, R the reference, f a feature, c a segment (base
phone), and t an (output, reference) pairing. R_sd is the reference's own
frame-level SD of feature f within segment c (natural-variation scaler).

Each output is paired against BOTH of its parents:
  - reference_type == "timbre": the accented target-speaker recording
  - reference_type == "style" : the english source recording

Emits the full long-format table with no aggregation.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "output.csv"
OUT = HERE / "rti_full.csv"
OUT_TABLE = HERE / "rti_by_model_feature.csv"
FIG_DIR = HERE.parent / "figures"

FEATURES = [
    # corrected harmonic-amplitude RATIOS only, harmonic-based (no formant-amplitude tilts)
    "H1H2c", "H2H4c", "H42Kc", "H2KH5Kc",
    # harmonics-to-noise ratios
    "HNR05", "HNR15", "HNR25", "HNR35",
    # snack formants + f0
    "sF1", "sF2", "sF3", "sF4", "sF0",
    # periodicity / energy
    "CPP", "Energy",
]

# ---- filename parsing -------------------------------------------------------
RE_COMMON = re.compile(r"^(openvoice|seed_vc)_(top5|bottom5)_(output|reference|source)_sentence([12])_(.+)\.mat$")
RE_OUTPUT = re.compile(r"^timbre-(.+?)__source-(.+?)__(?:openvoice|seed_vc)_s[12]$")
RE_REFSRC = re.compile(r"^(.+)_s[12]$")


def parse(fname: str):
    m = RE_COMMON.match(fname)
    if not m:
        return None
    model, rank, role, sent, rest = m.groups()
    info = {"filename": fname, "model": model, "rank": rank, "role": role, "sentence": sent}
    if role == "output":
        mo = RE_OUTPUT.match(rest)
        if not mo:
            return None
        info["timbre_id"] = mo.group(1)
        info["source_id"] = mo.group(2)
    else:  # reference or source
        mr = RE_REFSRC.match(rest)
        info["speaker_id"] = mr.group(1) if mr else rest
    return info


def base_phone(label: str) -> str:
    # strip word-context suffix: AA1_call -> AA1, R0_from -> R0, keep stress digit
    return str(label).split("_")[0]


def main():
    df = pd.read_csv(SRC, low_memory=False)
    df["seg"] = df["Label"].map(base_phone)

    meta = {fn: parse(fn) for fn in df["Filename"].unique()}
    unparsed = [fn for fn, v in meta.items() if v is None]
    if unparsed:
        print(f"WARNING: {len(unparsed)} filenames did not parse, e.g. {unparsed[:3]}")

    # per-recording, per-segment feature MEAN (for O and R) and SD (for R denom)
    grp = df.groupby(["Filename", "seg"])
    means = grp[FEATURES].mean()          # nan-aware
    sds = grp[FEATURES].std(ddof=1)       # sample SD across frames

    # per-feature denominator FLOOR: light insurance so a near-zero reference
    # SD can't detonate RTI. Computed as the 5th percentile of that feature's
    # reference-segment SDs.
    ref_files = {v["filename"] for v in meta.values()
                 if v and v["role"] in ("reference", "source")}
    ref_sds = sds.loc[sds.index.get_level_values(0).isin(ref_files)]
    floors = {}
    for f in FEATURES:
        p05 = ref_sds[f].quantile(0.05)
        med = ref_sds[f].median()
        # if even the 5th pctile collapses to ~0, fall back to the median
        floors[f] = p05 if (pd.notna(p05) and p05 > 1e-6) else (med if pd.notna(med) else 0.0)
    print("per-feature denominator floors (5th pctile of reference SDs):")
    for f in FEATURES:
        print(f"  {f:9s} {floors[f]:.4f}")

    # lookup tables for parents, keyed by (sentence, speaker_id)
    refs, srcs = {}, {}
    for fn, v in meta.items():
        if v is None:
            continue
        if v["role"] == "reference":
            refs.setdefault((v["sentence"], v["speaker_id"]), []).append(v)
        elif v["role"] == "source":
            srcs.setdefault((v["sentence"], v["speaker_id"]), []).append(v)

    def pick(candidates, model, rank):
        # prefer same model+rank prefix; else fall back to any (same speaker recording)
        for c in candidates:
            if c["model"] == model and c["rank"] == rank:
                return c
        return candidates[0]

    rows = []
    outputs = [v for v in meta.values() if v and v["role"] == "output"]
    for o in outputs:
        o_fn = o["filename"]
        if o_fn not in means.index.get_level_values(0):
            continue
        o_mean = means.loc[o_fn]
        pairings = [
            ("timbre", refs.get((o["sentence"], o["timbre_id"]))),
            ("style", srcs.get((o["sentence"], o["source_id"]))),
        ]
        for ref_type, cands in pairings:
            if not cands:
                print(f"WARNING: no {ref_type} reference for {o_fn}")
                continue
            r = pick(cands, o["model"], o["rank"])
            r_fn = r["filename"]
            r_mean = means.loc[r_fn]
            r_sd = sds.loc[r_fn]
            # segments present in BOTH output and this reference
            common = o_mean.index.intersection(r_mean.index)
            for seg in common:
                for f in FEATURES:
                    om = o_mean.at[seg, f]
                    rm = r_mean.at[seg, f]
                    rs = r_sd.at[seg, f]
                    raw_diff = om - rm
                    # RTI is a non-negative distance: numerator is |O - R|
                    valid = rs is not None and not np.isnan(rs) and rs != 0
                    rti = abs(raw_diff) / rs if valid else np.nan
                    # floored denominator: never smaller than the feature's floor
                    denom_f = max(rs, floors[f]) if (rs is not None and not np.isnan(rs)) else floors[f]
                    rti_fl = abs(raw_diff) / denom_f if denom_f and not np.isnan(raw_diff) else np.nan
                    rows.append({
                        "model": o["model"], "rank": o["rank"], "sentence": o["sentence"],
                        "pair_id": o_fn, "timbre_id": o["timbre_id"], "source_id": o["source_id"],
                        "reference_type": ref_type,
                        "reference_id": r.get("speaker_id"), "reference_file": r_fn,
                        "segment": seg, "feature": f,
                        "O_mean": om, "R_mean": rm, "R_sd": rs, "raw_diff": raw_diff,
                        "R_sd_floored": denom_f, "RTI": rti, "RTI_floored": rti_fl,
                    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out):,} rows -> {OUT}")
    print(f"outputs paired: {out['pair_id'].nunique()}  segments: {out['segment'].nunique()}  features: {out['feature'].nunique()}")
    print(f"RTI defined (non-NaN): {out['RTI'].notna().sum():,} / {len(out):,}")

    write_model_feature_table(out)


def write_model_feature_table(out: pd.DataFrame):
    """Collapse RTI to one value per (rank, reference_type, model, feature).

    RTI is already a unitless, standardized distance (scaled by the reference's
    within-segment SD), so it pools across segments directly -- formants need no
    special handling. Reported as median (robust to the right-skewed tail) and
    mean, over all segments, pairings and sentences, split by rank.
    """
    df = out[out["RTI_floored"].notna()]
    tab = (df.groupby(["rank", "reference_type", "model", "feature"])["RTI_floored"]
           .agg(median="median", mean="mean", n="count").reset_index())
    tab.to_csv(OUT_TABLE, index=False)
    for rk in ["top5", "bottom5"]:
        for rt in ["timbre", "style"]:
            piv = (tab[(tab["rank"] == rk) & (tab["reference_type"] == rt)]
                   .pivot(index="feature", columns="model", values="median")
                   .reindex(FEATURES).round(2))
            print(f"\nmedian RTI -- {rk} / {rt}:")
            print(piv.to_string())
    print(f"\nwrote {OUT_TABLE}")
    plot_model_feature_bars(tab)


def plot_model_feature_bars(tab: pd.DataFrame):
    """One grouped-bar figure per rank; features on x, openvoice vs seed_vc bars,
    timbre and style stacked as two panels. Dashed line at RTI=1 marks the
    reference's natural variation. Outlier bars are capped and value-labeled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    colors = {"openvoice": "#4E79A7", "seed_vc": "#F28E2B"}   # CVD-safe blue/orange
    caps = {"timbre": 2.2, "style": 4.0}
    x = np.arange(len(FEATURES))
    w = 0.38
    for rk in ["top5", "bottom5"]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        for ax, rt in zip(axes, ["timbre", "style"]):
            piv = (tab[(tab["rank"] == rk) & (tab["reference_type"] == rt)]
                   .pivot(index="feature", columns="model", values="median").reindex(FEATURES))
            cap = caps[rt]
            for k, model in enumerate(["openvoice", "seed_vc"]):
                vals = piv[model].values.astype(float)
                ax.bar(x + (k - 0.5) * w, np.minimum(vals, cap), w,
                       label=model, color=colors[model])
                for xi, v in zip(x + (k - 0.5) * w, vals):
                    if v > cap:   # outlier clipped: annotate true value
                        ax.text(xi, cap, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
            ax.axhline(1.0, color="#666", lw=1, ls="--", zorder=0)
            ax.set_ylim(0, cap)
            ax.set_xticks(x)
            ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("median RTI")
            ax.set_title(f"{rt} ({'target' if rt == 'timbre' else 'source'})", fontsize=11)
            ax.legend(frameon=False, fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle(f"Median RTI by feature and system  ({rk};  dashed = natural variation)", fontsize=12)
        path = FIG_DIR / f"rti_by_feature_{rk}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figure -> {path}")


if __name__ == "__main__":
    main()
