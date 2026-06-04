"""
Compute resemblyzer cosine similarity between each timbre reference and its
model output, rank pairs, copy top/bottom 5 + their TextGrids to analysis_files/.
Produces a side-by-side bar chart of all rankings for both models.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from resemblyzer import VoiceEncoder, preprocess_wav

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "reference_wav"
OV_DIR = ROOT / "output" / "openvoice" / "textgrids_cleaned"
SVC_DIR = ROOT / "output" / "seed_vc" / "textgrids_cleaned"
OUT_DIR = ROOT / "analysis_files"

N_EXTREME = 5  # top-N and bottom-N to copy


def embed(encoder: VoiceEncoder, wav_path: Path) -> np.ndarray:
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def extract_ref_id(filename: str) -> str:
    """e.g. 'timbre-gujarati10__2292__source-...' → 'gujarati10__2292'"""
    m = re.match(r"timbre-(.+?)__source-", filename)
    return m.group(1) if m else None


def build_pairs(model_dir: Path):
    """Return list of (ref_wav, out_wav, textgrid_or_None)."""
    pairs = []
    for out_wav in sorted(model_dir.glob("*.wav")):
        ref_id = extract_ref_id(out_wav.name)
        if ref_id is None:
            continue
        ref_wav = REF_DIR / f"{ref_id}.wav"
        if not ref_wav.exists():
            print(f"  [warn] no reference for {out_wav.name}")
            continue
        tg = out_wav.with_suffix(".TextGrid")
        pairs.append((ref_wav, out_wav, tg if tg.exists() else None))
    return pairs


def rank_pairs(encoder: VoiceEncoder, pairs):
    results = []
    for ref_wav, out_wav, tg in pairs:
        ref_emb = embed(encoder, ref_wav)
        out_emb = embed(encoder, out_wav)
        sim = cosine(ref_emb, out_emb)
        results.append({"ref": ref_wav, "out": out_wav, "tg": tg, "sim": sim})
    results.sort(key=lambda x: x["sim"], reverse=True)
    return results


def copy_extreme(ranked, model_name: str):
    model_out = OUT_DIR / model_name
    if model_out.exists():
        shutil.rmtree(model_out)
    top = ranked[:N_EXTREME]
    bot = ranked[-N_EXTREME:]
    for tier, group in [("top5", top), ("bottom5", bot)]:
        dest = OUT_DIR / model_name / tier
        tg_dest = OUT_DIR / model_name / f"{tier}_textgrids"
        dest.mkdir(parents=True, exist_ok=True)
        for item in group:
            shutil.copy2(item["ref"], dest / item["ref"].name)
            shutil.copy2(item["out"], dest / item["out"].name)
            if item["tg"]:
                tg_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item["tg"], tg_dest / item["tg"].name)


def plot_rankings(ov_ranked, svc_ranked):
    fig = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    for ax_idx, (ranked, title) in enumerate(
        [(ov_ranked, "OpenVoice"), (svc_ranked, "SeedVC")]
    ):
        ax = fig.add_subplot(gs[ax_idx])
        labels = [r["out"].stem.replace("timbre-", "").split("__")[0] for r in ranked]
        sims = [r["sim"] for r in ranked]
        n = len(sims)
        colors = []
        for i, s in enumerate(sims):
            if i < N_EXTREME:
                colors.append("#2ecc71")   # top-5 green
            elif i >= n - N_EXTREME:
                colors.append("#e74c3c")   # bottom-5 red
            else:
                colors.append("#95a5a6")   # middle grey

        bars = ax.barh(range(n), sims, color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()  # rank 1 at top
        ax.set_xlabel("Cosine Similarity")
        ax.set_title(f"{title}\nSpeaker Similarity Ranking", fontsize=11)
        ax.set_xlim(0, 1.05)

        # annotate sim value on each bar
        for i, (bar, s) in enumerate(zip(bars, sims)):
            ax.text(s + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{s:.3f}", va="center", fontsize=6)

    # legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color="#2ecc71", label=f"Top {N_EXTREME}"),
        Patch(color="#e74c3c", label=f"Bottom {N_EXTREME}"),
        Patch(color="#95a5a6", label="Middle"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout()
    out_path = OUT_DIR / "similarity_rankings.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")
    return out_path


def plot_grouped_bars(ov_ranked, svc_ranked):
    ov_by_id = {extract_ref_id(r["out"].name): r["sim"] for r in ov_ranked}
    svc_by_id = {extract_ref_id(r["out"].name): r["sim"] for r in svc_ranked}
    def lang(k):
        m = re.match(r"[a-z]+", k)
        return m.group() if m else k

    shared = sorted(set(ov_by_id) & set(svc_by_id), key=lambda k: (lang(k), k))

    labels = [k for k in shared]
    ov_sims = [ov_by_id[k] for k in shared]
    svc_sims = [svc_by_id[k] for k in shared]

    x = np.arange(len(shared))
    width = 0.4

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x - width / 2, ov_sims,  width, label="OpenVoice", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, svc_sims, width, label="SeedVC",    color="#e67e22", alpha=0.85)

    # language group shading
    langs = [lang(k) for k in shared]
    shade = False
    i = 0
    while i < len(langs):
        j = i
        while j < len(langs) and langs[j] == langs[i]:
            j += 1
        if shade:
            ax.axvspan(i - 0.5, j - 0.5, color="grey", alpha=0.08, zorder=0)
        ax.text((i + j - 1) / 2, ax.get_ylim()[0] + 0.002,
                langs[i].capitalize(), ha="center", fontsize=8, color="grey")
        shade = not shade
        i = j

    ax.set_xticks(x)
    ax.set_xticklabels([k.split("__")[0] for k in labels],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, len(shared) - 0.5)

    fig.tight_layout()
    out_path = OUT_DIR / "grouped_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading VoiceEncoder…")
    encoder = VoiceEncoder()

    print("Building OpenVoice pairs…")
    ov_pairs = build_pairs(OV_DIR)
    print(f"  {len(ov_pairs)} pairs found")

    print("Building SeedVC pairs…")
    svc_pairs = build_pairs(SVC_DIR)
    print(f"  {len(svc_pairs)} pairs found")

    print("Computing OpenVoice similarities…")
    ov_ranked = rank_pairs(encoder, ov_pairs)

    print("Computing SeedVC similarities…")
    svc_ranked = rank_pairs(encoder, svc_pairs)

    print("Copying extreme pairs…")
    copy_extreme(ov_ranked, "openvoice")
    copy_extreme(svc_ranked, "seed_vc")

    print("Plotting…")
    plot_rankings(ov_ranked, svc_ranked)
    plot_grouped_bars(ov_ranked, svc_ranked)

    # print summary tables
    for name, ranked in [("OpenVoice", ov_ranked), ("SeedVC", svc_ranked)]:
        print(f"\n{'='*55}")
        print(f"{name} — full ranking (best → worst)")
        print(f"{'='*55}")
        for rank, item in enumerate(ranked, 1):
            label = item["out"].stem.replace("timbre-", "").split("__source-")[0]
            print(f"  {rank:>2}. {label:<30} sim={item['sim']:.4f}")

    print("\nDone. Files written to:", OUT_DIR)


if __name__ == "__main__":
    main()
