#!/usr/bin/env python3
"""run_zstts.py

Single entrypoint to run zero-shot accent conversion pipelines:

1) OpenVoice (source_wav -> OpenVoice v2 tone color converter)
	- Source/content audio: each WAV in ./source_wav/
	- Timbre reference: each WAV in ./reference_wav/
	- Only runs pairs where genders match:
	   * timbre gender: data/accent_archive_metadata.csv (age_sex contains male/female)
	   * source gender: data/english_source_speakers.csv (age_sex or sex)
	- Writes:
	   output/openvoice/timbre-<reference_stem>__source-<source_stem>__openvoice.wav

2) SeedVC (style/source_wav -> timbre/reference_wav)
	- Source/style audio: each WAV in ./source_wav/
	- Target/timbre reference: each WAV in ./reference_wav/
	- Only runs pairs where genders match (same CSVs as OpenVoice)
	- Writes:
	   output/seedvc/timbre-<reference_stem>__style-<source_stem>__seedvc.wav

This script does NOT modify anything under ./OpenVoice/.
SeedVC is invoked via ./seed-vc/inference.py inside the configured conda env.

Run:
  python src/run_zstts.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Gender = Literal["male", "female", "unknown"]


@dataclass(frozen=True)
class Inputs:
	reference_wavs: list[Path]
	source_wavs: list[Path]


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _sanitize_filename(s: str) -> str:
	s = re.sub(r"\s+", "_", (s or "").strip())
	s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
	return s.strip("._-") or "out"


def _parse_gender(value: str) -> Gender:
	s = (value or "").strip().lower()
	if "female" in s:
		return "female"
	if "male" in s:
		return "male"
	return "unknown"


def list_wavs(folder: Path) -> list[Path]:
	folder = folder.expanduser().resolve()
	if not folder.exists():
		raise FileNotFoundError(f"Missing folder: {folder}")
	if not folder.is_dir():
		raise NotADirectoryError(f"Not a directory: {folder}")
	wavs = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
	if not wavs:
		raise FileNotFoundError(f"No .wav files found under: {folder}")
	return wavs


def load_gender_map_from_metadata_csv(metadata_csv: Path) -> dict[str, Gender]:
	"""reference mp3 stem (e.g. gujarati1__198) -> gender."""

	metadata_csv = metadata_csv.expanduser().resolve()
	if not metadata_csv.exists():
		raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")

	out: dict[str, Gender] = {}
	with metadata_csv.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			mp3_file = (row.get("mp3_file") or "").strip()
			stem = Path(mp3_file).stem if mp3_file else ""
			if not stem:
				continue
			out[stem] = _parse_gender(row.get("age_sex") or "")
	return out


def load_gender_map_from_english_sources_csv(english_sources_csv: Path) -> dict[str, Gender]:
	"""source mp3 stem (e.g. english137__509) -> gender."""

	english_sources_csv = english_sources_csv.expanduser().resolve()
	if not english_sources_csv.exists():
		raise FileNotFoundError(f"Missing english source speakers CSV: {english_sources_csv}")

	out: dict[str, Gender] = {}
	with english_sources_csv.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			mp3_file = (row.get("mp3_file") or "").strip()
			stem = Path(mp3_file).stem if mp3_file else ""
			if not stem:
				continue

			# Prefer explicit sex column if present, else fall back to age_sex.
			gender = _parse_gender(row.get("sex") or "")
			if gender == "unknown":
				gender = _parse_gender(row.get("age_sex") or "")
			out[stem] = gender
	return out


def _conda_python_argv(env_name: str) -> list[str]:
	return ["conda", "run", "-n", env_name, "--no-capture-output", "python"]


def _quote(argv: list[str]) -> str:
	return " ".join(shlex.quote(a) for a in argv)


def _run(argv: list[str], *, cwd: Path) -> None:
	print("\n[subprocess] cwd:", str(cwd))
	print("[subprocess] argv:", _quote(argv))
	subprocess.run(argv, cwd=str(cwd), check=True)


def ensure_dirs(output_root: Path) -> dict[str, Path]:
	output_root.mkdir(parents=True, exist_ok=True)
	openvoice_out = output_root / "openvoice"
	seedvc_out = output_root / "seed_vc"
	openvoice_out.mkdir(parents=True, exist_ok=True)
	seedvc_out.mkdir(parents=True, exist_ok=True)
	return {
		"output_root": output_root,
		"openvoice": openvoice_out,
		"seedvc": seedvc_out,
	}


def _move_single_wav(*, from_dir: Path, to_path: Path) -> None:
	from_dir = from_dir.expanduser().resolve()
	to_path = to_path.expanduser().resolve()
	if not from_dir.exists():
		raise FileNotFoundError(f"Missing expected SeedVC output directory: {from_dir}")
	wavs = [p for p in from_dir.glob("*.wav") if p.is_file()]
	if not wavs:
		raise FileNotFoundError(f"SeedVC did not produce any .wav under: {from_dir}")
	produced = max(wavs, key=lambda p: p.stat().st_mtime)
	to_path.parent.mkdir(parents=True, exist_ok=True)
	shutil.move(str(produced), str(to_path))


def orchestrate(
	*,
	openvoice_env: str,
	seedvc_env: str,
	reference_wav_dir: Path,
	source_wav_dir: Path,
	reference_metadata_csv: Path,
	english_sources_csv: Path,
	output_root: Path,
	seedvc_diffusion_steps: int = 30,
	seedvc_length_adjust: float = 1.0,
	seedvc_inference_cfg_rate: float = 0.7,
	seedvc_f0_condition: bool = False,
	seedvc_auto_f0_adjust: bool = False,
	seedvc_semi_tone_shift: int = 0,
	seedvc_fp16: bool = True,
	seedvc_checkpoint: str | None = None,
	seedvc_config: str | None = None,
	run_openvoice: bool = True,
	run_seedvc: bool = True,
) -> None:
	repo_root = _repo_root()
	dirs = ensure_dirs(output_root)

	reference_wavs = list_wavs(reference_wav_dir)
	source_wavs = list_wavs(source_wav_dir)
	inputs = Inputs(reference_wavs=reference_wavs, source_wavs=source_wavs)

	timbre_gender = load_gender_map_from_metadata_csv(reference_metadata_csv)
	style_gender = load_gender_map_from_english_sources_csv(english_sources_csv)

	self_script = Path(__file__).resolve()

	# ---------
	# OpenVoice
	# ---------
	if run_openvoice:
		# Convert each existing source wav into each reference speaker (gender-matched).
		for ref_wav in inputs.reference_wavs:
			ref_g = timbre_gender.get(ref_wav.stem, "unknown")
			if ref_g == "unknown":
				print(f"[OpenVoice] skip timbre={ref_wav.stem} (unknown gender in metadata)")
				continue

			for src_wav in inputs.source_wavs:
				src_g = style_gender.get(src_wav.stem, "unknown")
				if src_g == "unknown":
					print(f"[OpenVoice] skip source={src_wav.stem} (unknown gender in english_source_speakers)")
					continue
				if src_g != ref_g:
					print(
						f"[OpenVoice] skip pair (gender mismatch): timbre={ref_wav.stem}({ref_g}) source={src_wav.stem}({src_g})"
					)
					continue

				argv = (
					_conda_python_argv(openvoice_env)
					+ [
						str(self_script),
						"--mode",
						"openvoice-child",
						"--reference-wav",
						str(ref_wav),
						"--source-wav",
						str(src_wav),
						"--output-root",
						str(dirs["output_root"]),
					]
				)
				print(f"\n[OpenVoice] timbre={ref_wav.stem}({ref_g}) source={src_wav.stem}({src_g})")
				_run(argv, cwd=repo_root)

	# ---------
	# SeedVC (gender-matched pairs only)
	# ---------
	if not run_seedvc:
		return

	for ref_wav in inputs.reference_wavs:
		ref_g = timbre_gender.get(ref_wav.stem, "unknown")
		if ref_g == "unknown":
			print(f"[SeedVC] skip timbre={ref_wav.stem} (unknown gender in metadata)")
			continue

		for src_wav in inputs.source_wavs:
			src_g = style_gender.get(src_wav.stem, "unknown")
			if src_g == "unknown":
				print(f"[SeedVC] skip style={src_wav.stem} (unknown gender in english_source_speakers)")
				continue

			if src_g != ref_g:
				print(
					f"[SeedVC] skip pair (gender mismatch): timbre={ref_wav.stem}({ref_g}) style={src_wav.stem}({src_g})"
				)
				continue

			tmp_dir = (dirs["output_root"] / "_tmp_seedvc" / f"{ref_wav.stem}__{src_wav.stem}").resolve()
			if tmp_dir.exists():
				shutil.rmtree(tmp_dir)
			tmp_dir.mkdir(parents=True, exist_ok=True)

			seed_vc_root = (repo_root / "seed-vc").resolve()
			inference_py = seed_vc_root / "inference.py"
			if not inference_py.exists():
				raise FileNotFoundError(f"Missing SeedVC inference script: {inference_py}")

			argv = (
				_conda_python_argv(seedvc_env)
				+ [
					str(inference_py.name),
					"--source",
					str(src_wav),
					"--target",
					str(ref_wav),
					"--output",
					str(tmp_dir),
					"--diffusion-steps",
					str(int(seedvc_diffusion_steps)),
					"--length-adjust",
					str(float(seedvc_length_adjust)),
					"--inference-cfg-rate",
					str(float(seedvc_inference_cfg_rate)),
					"--f0-condition",
					"True" if seedvc_f0_condition else "False",
					"--auto-f0-adjust",
					"True" if seedvc_auto_f0_adjust else "False",
					"--semi-tone-shift",
					str(int(seedvc_semi_tone_shift)),
					"--fp16",
					"True" if seedvc_fp16 else "False",
				]
			)
			if seedvc_checkpoint:
				argv += ["--checkpoint", str(seedvc_checkpoint)]
			if seedvc_config:
				argv += ["--config", str(seedvc_config)]
			print(f"\n[SeedVC] timbre={ref_wav.stem}({ref_g}) style={src_wav.stem}({src_g})")
			try:
				_run(argv, cwd=seed_vc_root)
				out_wav = dirs["seedvc"] / (
					f"timbre-{_sanitize_filename(ref_wav.stem)}__source-{_sanitize_filename(src_wav.stem)}__seed_vc.wav"
				)
				_move_single_wav(from_dir=tmp_dir, to_path=out_wav)
				print(f"[seedvc] Wrote: {out_wav}")
			except subprocess.CalledProcessError as e:
				print(
					"[SeedVC] ERROR (continuing). Pair failed: "
					f"timbre={ref_wav.stem} style={src_wav.stem} exit_code={e.returncode}",
					file=sys.stderr,
				)


def openvoice_child(*, reference_wav: Path, source_wav: Path, output_root: Path) -> None:
	"""Run one OpenVoice conversion (under the openvoice conda env)."""

	repo_root = _repo_root()
	openvoice_root = repo_root / "OpenVoice"
	if not openvoice_root.exists():
		raise FileNotFoundError(f"Missing OpenVoice directory: {openvoice_root}")

	reference_wav = reference_wav.expanduser().resolve()
	if not reference_wav.exists():
		raise FileNotFoundError(f"Missing reference wav: {reference_wav}")

	source_wav = source_wav.expanduser().resolve()
	if not source_wav.exists():
		raise FileNotFoundError(f"Missing source wav: {source_wav}")

	dirs = ensure_dirs(output_root.expanduser().resolve())

	# Make vendored OpenVoice importable.
	sys.path.insert(0, str(openvoice_root))

	import torch

	from openvoice import se_extractor  # type: ignore[import-not-found]
	from openvoice.api import ToneColorConverter  # type: ignore[import-not-found]

	ckpt_converter_dir = openvoice_root / "checkpoints_v2" / "converter"
	config_json = ckpt_converter_dir / "config.json"
	checkpoint_pth = ckpt_converter_dir / "checkpoint.pth"
	if not config_json.exists() or not checkpoint_pth.exists():
		raise FileNotFoundError(
			"OpenVoice v2 converter checkpoints missing. Expected:\n"
			f"  - {config_json}\n"
			f"  - {checkpoint_pth}"
		)

	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	source_audio = source_wav

	# 2) OpenVoice converter
	converter = ToneColorConverter(str(config_json), device=device)
	converter.load_ckpt(str(checkpoint_pth))

	# Some forks cache a whisper model; reset when present.
	if hasattr(se_extractor, "model"):
		se_extractor.model = None

	source_se, _ = se_extractor.get_se(str(source_audio), converter, vad=False)
	target_se, _ = se_extractor.get_se(str(reference_wav), converter, vad=False)

	out_name = (
		f"timbre-{_sanitize_filename(reference_wav.stem)}__source-{_sanitize_filename(source_wav.stem)}__openvoice.wav"
	)
	out_wav = dirs["openvoice"] / out_name
	try:
		converter.convert(str(source_audio), source_se, target_se, str(out_wav))
	except TypeError:
		converter.convert(
			audio_src_path=str(source_audio),
			src_se=source_se,
			tgt_se=target_se,
			output_path=str(out_wav),
		)

	if not out_wav.exists() or out_wav.stat().st_size == 0:
		raise RuntimeError("[openvoice] Conversion did not create expected output wav")
	print(f"[openvoice] Wrote converted: {out_wav}")


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Run OpenVoice + SeedVC (single entrypoint)")

	p.add_argument(
		"--mode",
		choices=("run", "openvoice-only", "seedvc-only", "openvoice-child"),
		default="run",
		help="Parent orchestrator (run) or child mode (runs inside model conda env)",
	)

	p.add_argument("--openvoice-env", default=os.environ.get("OPENVOICE_ENV", "openvoice"))
	p.add_argument("--seedvc-env", default=os.environ.get("SEEDVC_ENV", "seed"))

	# SeedVC quality knobs (passed through to seed-vc/inference.py)
	p.add_argument(
		"--seedvc-diffusion-steps",
		type=int,
		default=30,
		help="SeedVC diffusion steps. 30–50 best quality; 4–10 fastest.",
	)
	p.add_argument(
		"--seedvc-length-adjust",
		type=float,
		default=1.0,
		help="SeedVC length adjustment factor (<1 faster, >1 slower).",
	)
	p.add_argument(
		"--seedvc-inference-cfg-rate",
		type=float,
		default=0.7,
		help="SeedVC inference CFG rate (subtle effect).",
	)
	def _str2bool(v: str) -> bool:
		return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

	p.add_argument(
		"--seedvc-f0-condition",
		type=_str2bool,
		default=False,
		help="SeedVC f0 conditioning (mainly for singing voice conversion).",
	)
	p.add_argument(
		"--seedvc-auto-f0-adjust",
		type=_str2bool,
		default=False,
		help="SeedVC auto F0 adjust (normally not used for singing VC).",
	)
	p.add_argument(
		"--seedvc-semi-tone-shift",
		type=int,
		default=0,
		help="SeedVC pitch shift in semitones (singing VC).",
	)
	p.add_argument(
		"--seedvc-fp16",
		type=_str2bool,
		default=True,
		help="SeedVC fp16 flag. On Apple Silicon (mps), True/False may behave similarly.",
	)
	p.add_argument(
		"--seedvc-checkpoint",
		default=None,
		help="Optional SeedVC checkpoint path (defaults to auto-download).",
	)
	p.add_argument(
		"--seedvc-config",
		default=None,
		help="Optional SeedVC config path (defaults to auto-download).",
	)

	p.add_argument("--reference-wav-dir", default=str(_repo_root() / "reference_wav"))
	p.add_argument("--source-wav-dir", default=str(_repo_root() / "source_wav"))
	p.add_argument("--reference-metadata-csv", default=str(_repo_root() / "data" / "accent_archive_metadata.csv"))
	p.add_argument("--english-sources-csv", default=str(_repo_root() / "data" / "english_source_speakers.csv"))

	p.add_argument(
		"--output-root",
		default=str(_repo_root() / "output"),
		help="All outputs go under this folder (subfolders will be created)",
	)

	# Note: OpenVoice no longer uses MeloTTS; it converts source_wav -> reference_wav.

	# Child-only args
	p.add_argument("--reference-wav", help=argparse.SUPPRESS)
	p.add_argument("--source-wav", help=argparse.SUPPRESS)

	return p


def main() -> None:
	args = build_arg_parser().parse_args()
	output_root = Path(args.output_root).expanduser().resolve()

	if args.mode == "run":
		orchestrate(
			openvoice_env=args.openvoice_env,
			seedvc_env=args.seedvc_env,
			reference_wav_dir=Path(args.reference_wav_dir),
			source_wav_dir=Path(args.source_wav_dir),
			reference_metadata_csv=Path(args.reference_metadata_csv),
			english_sources_csv=Path(args.english_sources_csv),
			output_root=output_root,
			seedvc_diffusion_steps=args.seedvc_diffusion_steps,
			seedvc_length_adjust=args.seedvc_length_adjust,
			seedvc_inference_cfg_rate=args.seedvc_inference_cfg_rate,
			seedvc_f0_condition=args.seedvc_f0_condition,
			seedvc_auto_f0_adjust=args.seedvc_auto_f0_adjust,
			seedvc_semi_tone_shift=args.seedvc_semi_tone_shift,
			seedvc_fp16=args.seedvc_fp16,
			seedvc_checkpoint=(args.seedvc_checkpoint or None),
			seedvc_config=(args.seedvc_config or None),
			run_openvoice=True,
			run_seedvc=True,
		)
		return

	if args.mode == "openvoice-only":
		orchestrate(
			openvoice_env=args.openvoice_env,
			seedvc_env=args.seedvc_env,
			reference_wav_dir=Path(args.reference_wav_dir),
			source_wav_dir=Path(args.source_wav_dir),
			reference_metadata_csv=Path(args.reference_metadata_csv),
			english_sources_csv=Path(args.english_sources_csv),
			output_root=output_root,
			seedvc_diffusion_steps=args.seedvc_diffusion_steps,
			seedvc_length_adjust=args.seedvc_length_adjust,
			seedvc_inference_cfg_rate=args.seedvc_inference_cfg_rate,
			seedvc_f0_condition=args.seedvc_f0_condition,
			seedvc_auto_f0_adjust=args.seedvc_auto_f0_adjust,
			seedvc_semi_tone_shift=args.seedvc_semi_tone_shift,
			seedvc_fp16=args.seedvc_fp16,
			seedvc_checkpoint=(args.seedvc_checkpoint or None),
			seedvc_config=(args.seedvc_config or None),
			run_openvoice=True,
			run_seedvc=False,
		)
		return

	if args.mode == "seedvc-only":
		orchestrate(
			openvoice_env=args.openvoice_env,
			seedvc_env=args.seedvc_env,
			reference_wav_dir=Path(args.reference_wav_dir),
			source_wav_dir=Path(args.source_wav_dir),
			reference_metadata_csv=Path(args.reference_metadata_csv),
			english_sources_csv=Path(args.english_sources_csv),
			output_root=output_root,
			seedvc_diffusion_steps=args.seedvc_diffusion_steps,
			seedvc_length_adjust=args.seedvc_length_adjust,
			seedvc_inference_cfg_rate=args.seedvc_inference_cfg_rate,
			seedvc_f0_condition=args.seedvc_f0_condition,
			seedvc_auto_f0_adjust=args.seedvc_auto_f0_adjust,
			seedvc_semi_tone_shift=args.seedvc_semi_tone_shift,
			seedvc_fp16=args.seedvc_fp16,
			seedvc_checkpoint=(args.seedvc_checkpoint or None),
			seedvc_config=(args.seedvc_config or None),
			run_openvoice=False,
			run_seedvc=True,
		)
		return

	if args.mode == "openvoice-child":
		if not args.reference_wav or not args.source_wav:
			raise SystemExit("openvoice-child requires --reference-wav and --source-wav")
		openvoice_child(
			reference_wav=Path(args.reference_wav),
			source_wav=Path(args.source_wav),
			output_root=output_root,
		)
		return

	raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
	main()

