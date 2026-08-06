from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MFA_ENV_BIN = Path("/opt/anaconda3/envs/accent_conversion/bin")
MFA_BIN = str(MFA_ENV_BIN / "mfa")


def _find_transcript(corpus_dir: Path, stem: str) -> Path | None:
    for ext in (".lab", ".txt"):
        p = corpus_dir / f"{stem}{ext}"
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _find_transcript_for_wav(wav_path: Path) -> Path | None:
    """Find transcript next to a wav (same stem, .lab or .txt)."""
    return _find_transcript(wav_path.parent, wav_path.stem)


def _extract_source_stem_from_generated(generated_stem: str) -> str | None:
    """Parse stems like '...__source-english584__2197__openvoice' -> 'english584__2197'."""
    marker = "__source-"
    if marker not in generated_stem:
        return None
    tail = generated_stem.split(marker, 1)[1]
    parts = tail.split("__")
    if len(parts) < 2:
        return None
    return "__".join(parts[:-1])


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _run_mfa_align(*, corpus_dir: Path, output_dir: Path) -> None:
    cmd = [
        MFA_BIN,
        "align",
        str(corpus_dir),
        "english_us_arpa",  # dictionary model
        "english_us_arpa",  # acoustic model
        str(output_dir),
        "--clean",
        "--overwrite",
        "--single_speaker",
    ]

    env = os.environ.copy()
    env["PATH"] = str(MFA_ENV_BIN) + ":" + env.get("PATH", "")
    subprocess.run(cmd, check=True, env=env)



def run_align_incremental(
    corpus_dir: Path,
    output_dir: Path,
    *,
    skip_existing: bool = True,
    transcript_resolver: Callable[[str], Path | None] | None = None,
) -> None:
    corpus_dir = corpus_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[MFA] Corpus: {corpus_dir}")
    print(f"[MFA] Output TextGrids: {output_dir}")

    audio_files = sorted([p for p in corpus_dir.rglob("*.wav") if p.is_file()])
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in corpus_dir={corpus_dir}")

    missing_stems: list[str] = []
    skipped_existing = 0
    skipped_no_transcript = 0

    for wav_path in audio_files:
        stem = wav_path.stem
        out_grid = output_dir / f"{stem}.TextGrid"
        if skip_existing and out_grid.exists() and out_grid.stat().st_size > 0:
            skipped_existing += 1
            continue

        transcript = _find_transcript_for_wav(wav_path)
        if transcript is None and transcript_resolver is not None:
            try:
                transcript = transcript_resolver(stem)
            except Exception:
                transcript = None
        if transcript is None:
            skipped_no_transcript += 1
            continue

        missing_stems.append(stem)

    print(f"[MFA] wavs total: {len(audio_files)}")
    print(f"[MFA] will align (missing TextGrid): {len(missing_stems)}")
    if skip_existing:
        print(f"[MFA] skip existing TextGrid: {skipped_existing}")
    if skipped_no_transcript:
        print(f"[MFA] skip missing transcript (.lab/.txt): {skipped_no_transcript}")

    if not missing_stems:
        print("[MFA] Nothing to do; all TextGrids already exist.")
        return

    with tempfile.TemporaryDirectory(prefix="mfa_incremental_") as td:
        tmp_root = Path(td)
        tmp_corpus = tmp_root / "corpus"
        tmp_out = tmp_root / "out"
        tmp_corpus.mkdir(parents=True, exist_ok=True)
        tmp_out.mkdir(parents=True, exist_ok=True)

        stem_to_wav: dict[str, Path] = {}
        stem_to_transcript: dict[str, Path] = {}

        for wav_path in audio_files:
            stem = wav_path.stem
            if stem not in missing_stems:
                continue
            transcript = _find_transcript_for_wav(wav_path)
            if transcript is None and transcript_resolver is not None:
                try:
                    transcript = transcript_resolver(stem)
                except Exception:
                    transcript = None
            if transcript is None:
                continue
            stem_to_wav[stem] = wav_path
            stem_to_transcript[stem] = transcript

        for stem in missing_stems:
            wav_path = stem_to_wav.get(stem)
            transcript = stem_to_transcript.get(stem)
            if wav_path is None or transcript is None:
                continue

            _safe_link_or_copy(wav_path, tmp_corpus / wav_path.name)
            # Transcript in the mini-corpus must match the wav stem.
            _safe_link_or_copy(transcript, tmp_corpus / f"{stem}{transcript.suffix}")

        print(f"\n[MFA] Running incremental align on {len(missing_stems)} items...")
        _run_mfa_align(corpus_dir=tmp_corpus, output_dir=tmp_out)

        copied = 0
        for stem in missing_stems:
            candidates = list(tmp_out.rglob(f"{stem}.TextGrid"))
            if not candidates:
                print(f"[MFA] WARNING: missing expected TextGrid for {stem}")
                continue
            # Take the first match (should be unique for this mini-corpus).
            src_grid = candidates[0]
            dst_grid = output_dir / src_grid.name
            if skip_existing and dst_grid.exists() and dst_grid.stat().st_size > 0:
                continue
            shutil.copy2(src_grid, dst_grid)
            copied += 1

        print(f"[MFA] Wrote {copied} new TextGrids")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Montreal Forced Aligner (incremental)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (defaults to the accent_conversion repo root)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve()

    reference_wav = root / "reference_wav"
    reference_tg = root / "reference_textgrids"
    source_wav = root / "source_wav"
    source_tg = root / "source_textgrids"

    out_openvoice = root / "output" / "openvoice"
    out_openvoice_tg = out_openvoice / "textgrids"
    out_seed_vc = root / "output" / "seed_vc"
    out_seed_vc_tg = out_seed_vc / "textgrids"

    def resolve_generated_transcript(generated_stem: str) -> Path | None:
        source_stem = _extract_source_stem_from_generated(generated_stem)
        if source_stem is None:
            return None
        return _find_transcript(source_wav, source_stem)

    run_align_incremental(reference_wav, reference_tg, skip_existing=bool(args.skip_existing))
    run_align_incremental(source_wav, source_tg, skip_existing=bool(args.skip_existing))

    if out_openvoice.exists():
        run_align_incremental(
            out_openvoice,
            out_openvoice_tg,
            skip_existing=bool(args.skip_existing),
            transcript_resolver=resolve_generated_transcript,
        )

    if out_seed_vc.exists():
        run_align_incremental(
            out_seed_vc,
            out_seed_vc_tg,
            skip_existing=bool(args.skip_existing),
            transcript_resolver=resolve_generated_transcript,
        )

if __name__ == "__main__":
    main()