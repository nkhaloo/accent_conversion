#!/usr/bin/env python3
"""run_accent_conversion.py

One-command accent conversion experiment runner that calls:

1) OpenVoice (text -> speech in reference timbre)
2) Vevo (map reference timbre onto source style)

Key design goals (based on your request):
- Self-contained: this script is the *only* entrypoint you run.
- No dependency changes: it does not install/modify requirements.
- Separate environments: it can spawn subprocesses using different Python commands
  for OpenVoice vs Vevo.
- Outputs: everything is written under ./output/

Typical usage (recommended):

  # If you use conda envs named 'openvoice' and 'vevo':
  python run_accent_conversion.py \
    --openvoice-py "conda run -n openvoice python" \
    --vevo-py "conda run -n vevo python"

If you prefer, you can set env vars instead:
  export OPENVOICE_PY_CMD="conda run -n openvoice python"
  export VEVO_PY_CMD="conda run -n vevo python"
  python run_accent_conversion.py

Inputs default to:
- ./reference/  (first audio file inside)
- ./source/     (first audio file inside)

Notes:
- OpenVoice V2 needs checkpoints in OpenVoice/checkpoints_v2 (as per OpenVoice docs).
- Vevo will auto-download model artifacts via HuggingFace into Amphion/ckpts/Vevo.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


@dataclass(frozen=True)
class ResolvedInputs:
    reference_audio: Path
    source_audio: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _first_audio_file(path: Path) -> Path:
    """Accept either a file path or a directory containing audio."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    # Pick the first audio file in a stable order.
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTS:
            return candidate

    raise FileNotFoundError(
        f"No audio files found under directory: {path} (looked for {AUDIO_EXTS})"
    )


def _ensure_output_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_wav(
    *,
    input_audio: Path,
    output_wav: Path,
    target_sr: int = 24000,
) -> Path:
    """Convert arbitrary audio to WAV.

    Why this exists:
    - Your `source/` may contain `.mp3` etc, but Vevo inference is most robust when
      given `.wav` paths.

    How it works (no requirements changes):
    1) Tries `ffmpeg` if it exists on your system (fast + reliable).
    2) Falls back to a pure-Python path using `librosa` to decode + `torchaudio` to save.

    This function is called inside the model-specific child modes, so any optional
    python libraries only need to exist in that *model's* environment.
    """

    input_audio = input_audio.expanduser().resolve()
    output_wav = output_wav.expanduser().resolve()

    if input_audio.suffix.lower() == ".wav":
        return input_audio

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1) Prefer ffmpeg if present (does not require Python deps).
    try:
        argv = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            str(output_wav),
        ]
        subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if output_wav.exists() and output_wav.stat().st_size > 0:
            return output_wav
    except Exception:
        # Either ffmpeg isn't installed, or it failed on this file.
        pass

    # 2) Python fallback (requires librosa + torchaudio in the current environment).
    import numpy as np
    import torch
    import librosa
    import torchaudio

    y, _sr = librosa.load(str(input_audio), sr=target_sr, mono=True)
    if not isinstance(y, np.ndarray) or y.size == 0:
        raise RuntimeError(f"Failed to decode audio: {input_audio}")
    wav_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    torchaudio.save(str(output_wav), wav_tensor, target_sr)
    return output_wav


def _parse_cmd(cmd: str) -> list[str]:
    """Split a shell-like python command string into argv safely."""
    # Example: "conda run -n openvoice python" -> ["conda","run","-n","openvoice","python"]
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("Empty python command")
    return shlex.split(cmd)


def _conda_env_exists(env_name: str) -> bool:
    """Return True if `conda` is available and the env name appears in `conda env list`."""
    if shutil.which("conda") is None:
        return False
    try:
        out = subprocess.check_output(["conda", "env", "list"], text=True)
    except Exception:
        return False
    # Lines look like: "openvoice              /opt/anaconda3/envs/openvoice"
    return any(line.split()[:1] == [env_name] for line in out.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def _default_python_cmd_for_env(env_name: str) -> str | None:
    """Provide a sensible default python command for a conda env if available."""
    if _conda_env_exists(env_name):
        return f"conda run -n {env_name} python"
    return None


def _run_subprocess(argv: Sequence[str], cwd: Path) -> None:
    """Run and stream subprocess output; raise on non-zero exit."""
    print("\n[subprocess] cwd:", str(cwd))
    print("[subprocess] argv:", " ".join(shlex.quote(a) for a in argv))
    subprocess.run(argv, cwd=str(cwd), check=True)


def resolve_inputs(reference: Path, source: Path) -> ResolvedInputs:
    return ResolvedInputs(
        reference_audio=_first_audio_file(reference),
        source_audio=_first_audio_file(source),
    )


# ----------------------------
# Orchestration (parent mode)
# ----------------------------

def run_both_in_envs(
    *,
    self_script: Path,
    out_dir: Path,
    text: str,
    inputs: ResolvedInputs,
    openvoice_py_cmd: str,
    vevo_py_cmd: str,
) -> None:
    """Spawn two child processes, each under its own Python command."""

    # We call *this same script* in child mode. This keeps the experiment as
    # one file, while still letting each model run inside its own environment.

    # --- OpenVoice child ---
    openvoice_argv = (
        _parse_cmd(openvoice_py_cmd)
        + [
            str(self_script),
            "--mode",
            "openvoice",
            "--text",
            text,
            "--reference-audio",
            str(inputs.reference_audio),
            "--source-audio",
            str(inputs.source_audio),
            "--out-dir",
            str(out_dir),
        ]
    )
    _run_subprocess(openvoice_argv, cwd=_repo_root())

    # --- Vevo child ---
    vevo_argv = (
        _parse_cmd(vevo_py_cmd)
        + [
            str(self_script),
            "--mode",
            "vevo",
            "--text",
            text,
            "--reference-audio",
            str(inputs.reference_audio),
            "--source-audio",
            str(inputs.source_audio),
            "--out-dir",
            str(out_dir),
        ]
    )
    _run_subprocess(vevo_argv, cwd=_repo_root())


# ----------------------------
# OpenVoice implementation
# ----------------------------

def run_openvoice(
    *,
    out_dir: Path,
    text: str,
    reference_audio: Path,
) -> None:
    """Run OpenVoice V2: text -> MeloTTS wav -> tone-color conversion to reference timbre."""

    # IMPORTANT: We intentionally import OpenVoice/Melo deps *inside* this function.
    # That way, the parent/orchestrator mode stays dependency-free.

    repo_root = _repo_root()
    openvoice_root = repo_root / "OpenVoice"
    if not openvoice_root.exists():
        raise FileNotFoundError(f"Missing OpenVoice folder: {openvoice_root}")

    # Make local OpenVoice package importable even if it wasn't pip-installed.
    sys.path.insert(0, str(openvoice_root))

    # OpenVoice V2 checkpoints are required (per your request).
    ckpt_converter_dir = openvoice_root / "checkpoints_v2" / "converter"
    config_json = ckpt_converter_dir / "config.json"
    checkpoint_pth = ckpt_converter_dir / "checkpoint.pth"
    if not config_json.exists() or not checkpoint_pth.exists():
        raise FileNotFoundError(
            "OpenVoice V2 converter checkpoints not found. Expected files:\n"
            f"  - {config_json}\n"
            f"  - {checkpoint_pth}\n"
        )
    print(f"[openvoice] Using OpenVoice v2 converter checkpoints at: {ckpt_converter_dir}")

    # Now import the OpenVoice APIs.
    import torch

    from openvoice import se_extractor  # type: ignore[import-not-found]
    from openvoice.api import ToneColorConverter  # type: ignore[import-not-found]

    # MeloTTS is used as a convenient TTS front-end.
    # It is required for OpenVoice V2 usage and is also fine to use with V1 checkpoints.
    from melo.api import TTS

    _ensure_output_dir(out_dir)

    # Choose CPU on macOS unless CUDA is available.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) TTS from text -> raw wav (this provides the *content*).
    tts_raw_path = out_dir / "openvoice_tts_raw.wav"
    print("[openvoice] Loading MeloTTS model...")
    melo = TTS(language="EN", device="auto")

    speaker_ids = melo.hps.data.spk2id
    spk_key = "EN-Default" if "EN-Default" in speaker_ids else next(iter(speaker_ids.keys()))
    spk_id = speaker_ids[spk_key]

    print(f"[openvoice] Synthesizing text with speaker='{spk_key}' -> {tts_raw_path}")
    melo.tts_to_file(text, spk_id, str(tts_raw_path), speed=1.0)

    if not tts_raw_path.exists() or tts_raw_path.stat().st_size == 0:
        raise RuntimeError("[openvoice] MeloTTS did not create the expected WAV.")

    # 2) Load tone-color converter.
    print("[openvoice] Loading ToneColorConverter...")
    converter = ToneColorConverter(str(config_json), device=device)
    converter.load_ckpt(str(checkpoint_pth))

    # 3) Extract speaker embeddings (SE) for source (tts) and target (reference).
    # Note: some forks cache a Whisper model globally; clearing is often safer.
    if hasattr(se_extractor, "model"):
        se_extractor.model = None

    print("[openvoice] Extracting SOURCE embedding from TTS wav")
    source_se, _ = se_extractor.get_se(str(tts_raw_path), converter, vad=False)

    print("[openvoice] Extracting TARGET embedding from reference audio")
    target_se, _ = se_extractor.get_se(str(reference_audio), converter, vad=False)

    # 4) Convert.
    converted_path = out_dir / "openvoice_converted.wav"
    print(f"[openvoice] Converting -> {converted_path}")

    # OpenVoice forks differ slightly in convert() signature.
    # We attempt the most common call form first and fall back to keywords.
    try:
        converter.convert(str(tts_raw_path), source_se, target_se, str(converted_path))
    except TypeError:
        converter.convert(
            audio_src_path=str(tts_raw_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(converted_path),
        )

    if not converted_path.exists() or converted_path.stat().st_size == 0:
        raise RuntimeError("[openvoice] Conversion did not create the expected WAV.")

    print("[openvoice] Done")


# ----------------------------
# Vevo implementation
# ----------------------------

def run_vevo(
    *,
    out_dir: Path,
    reference_audio: Path,
    source_audio: Path,
) -> None:
    """Run Vevo style/timbre mapping.

    Your requested mapping:
    - timbre reference: ./reference/*
    - style reference:  ./source/*

    Practical interpretation (since Vevo takes audio, not text):
    - We use the *source audio* as the content to be re-synthesized.
    - We use the same source audio as the style reference (accent/style).
    - We use the reference audio as the timbre reference (speaker identity).

    This yields: reference speaker (timbre) speaking the linguistic content of source,
    in the style/accent of source.
    """

    repo_root = _repo_root()
    amphion_root = repo_root / "Amphion"
    if not amphion_root.exists():
        raise FileNotFoundError(f"Missing Amphion folder: {amphion_root}")

    # Make Amphion importable even if it wasn't pip-installed.
    sys.path.insert(0, str(amphion_root))

    # We keep imports inside this function so parent mode stays stdlib-only.
    import torch
    from huggingface_hub import snapshot_download

    from models.vc.vevo.vevo_utils import (  # type: ignore[import-not-found]
        VevoInferencePipeline,
        save_audio,
    )

    _ensure_output_dir(out_dir)

    # Convert the inputs to WAVs (saved into output/) so you can inspect them,
    # and so Vevo always sees `.wav` paths.
    # This satisfies: "the source file should also be converted to .wav".
    source_wav = ensure_wav(
        input_audio=source_audio,
        output_wav=out_dir / "source_converted.wav",
        target_sr=24000,
    )
    reference_wav = ensure_wav(
        input_audio=reference_audio,
        output_wav=out_dir / "reference_converted.wav",
        target_sr=24000,
    )

    # IMPORTANT: Vevo scripts expect to be run under the Amphion root so
    # relative paths like ./models/... resolve.
    os.chdir(amphion_root)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Download model artifacts (HuggingFace) into Amphion/ckpts/Vevo.
    # This does not change requirements; it only populates checkpoints.

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq32/*"],
    )
    content_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
    )

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"],
    )
    ar_cfg_path = "./models/vc/vevo/config/Vq32ToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Build inference pipeline.
    inference_pipeline = VevoInferencePipeline(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    # Run style + timbre-controlled conversion.
    # src_wav_path: provides the linguistic content
    # style_ref_wav_path: provides style/accent
    # timbre_ref_wav_path: provides speaker identity
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=str(source_wav),
        src_text=None,
        style_ref_wav_path=str(source_wav),
        timbre_ref_wav_path=str(reference_wav),
    )

    out_path = out_dir / "vevo_style_timbre_mapped.wav"
    save_audio(gen_audio, output_path=str(out_path))

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("[vevo] Inference did not create the expected WAV.")

    print("[vevo] Done")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run OpenVoice + Vevo accent conversion experiment")

    p.add_argument(
        "--mode",
        choices=("both", "openvoice", "vevo"),
        default="both",
        help="Run both models (parent/orchestrator) or run a single model (child mode)",
    )

    p.add_argument(
        "--text",
        default="I like to eat cake for my birthday",
        help="Text used for OpenVoice TTS input",
    )

    p.add_argument(
        "--reference",
        default=str(_repo_root() / "reference"),
        help="Reference speaker audio file OR a directory containing it",
    )

    p.add_argument(
        "--source",
        default=str(_repo_root() / "source"),
        help="Style reference audio file OR a directory containing it",
    )

    # These are passed internally when parent -> child. Keep them hidden in help.
    p.add_argument("--reference-audio", help=argparse.SUPPRESS)
    p.add_argument("--source-audio", help=argparse.SUPPRESS)

    p.add_argument(
        "--out-dir",
        default=str(_repo_root() / "output"),
        help="Output directory",
    )

    p.add_argument(
        "--openvoice-py",
        default=os.environ.get("OPENVOICE_PY_CMD", ""),
        help="Python command for OpenVoice env (e.g. 'conda run -n openvoice python')",
    )

    p.add_argument(
        "--vevo-py",
        default=os.environ.get("VEVO_PY_CMD", ""),
        help="Python command for Vevo env (e.g. 'conda run -n vevo python')",
    )

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = _ensure_output_dir(Path(args.out_dir).expanduser().resolve())

    # Resolve actual audio files from ./reference and ./source.
    if args.reference_audio and args.source_audio:
        # Child mode: parent already picked explicit files.
        inputs = ResolvedInputs(
            reference_audio=Path(args.reference_audio).expanduser().resolve(),
            source_audio=Path(args.source_audio).expanduser().resolve(),
        )
    else:
        inputs = resolve_inputs(Path(args.reference), Path(args.source))

    print("[inputs] reference_audio:", inputs.reference_audio)
    print("[inputs] source_audio:", inputs.source_audio)
    print("[outputs] out_dir:", out_dir)

    if args.mode == "both":
        # Parent/orchestrator mode.
        # If the user didn't pass explicit commands, try to auto-detect the standard
        # conda env names. This makes VS Code "Run Python File" work out-of-the-box.

        openvoice_py = args.openvoice_py or _default_python_cmd_for_env("openvoice")
        vevo_py = args.vevo_py or _default_python_cmd_for_env("vevo")

        if not openvoice_py:
            raise SystemExit(
                "Missing --openvoice-py (or OPENVOICE_PY_CMD). Example:\n"
                "  --openvoice-py 'conda run -n openvoice python'\n\n"
                "Tip: if you have a conda env actually named 'openvoice', this script can auto-detect it,\n"
                "but it looks like conda isn't available or that env name doesn't exist."
            )
        if not vevo_py:
            raise SystemExit(
                "Missing --vevo-py (or VEVO_PY_CMD). Example:\n"
                "  --vevo-py 'conda run -n vevo python'\n\n"
                "Tip: if you have a conda env actually named 'vevo', this script can auto-detect it,\n"
                "but it looks like conda isn't available or that env name doesn't exist."
            )

        run_both_in_envs(
            self_script=Path(__file__).resolve(),
            out_dir=out_dir,
            text=args.text,
            inputs=inputs,
            openvoice_py_cmd=openvoice_py,
            vevo_py_cmd=vevo_py,
        )
        return

    if args.mode == "openvoice":
        # Child mode: run OpenVoice only.
        run_openvoice(out_dir=out_dir, text=args.text, reference_audio=inputs.reference_audio)
        return

    if args.mode == "vevo":
        # Child mode: run Vevo only.
        run_vevo(out_dir=out_dir, reference_audio=inputs.reference_audio, source_audio=inputs.source_audio)
        return

    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
