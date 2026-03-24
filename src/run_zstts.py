#!/usr/bin/env python3
"""run_zstts.py

Single entrypoint to run both zero-shot accent conversion pipelines:

1) OpenVoice (source_wav -> OpenVoice v2 tone color converter)
	- Source/content audio: each WAV in ./source_wav/
	- Timbre reference: each WAV in ./reference_wav/
	- Only runs pairs where genders match:
	   * timbre gender: data/accent_archive_metadata.csv (age_sex contains male/female)
	   * source gender: data/english_source_speakers.csv (age_sex or sex)
	- Writes:
	   output/openvoice/timbre-<reference_stem>__source-<source_stem>__openvoice.wav

2) VevoStyle (Amphion Vevo)
   - Style/content reference: each WAV in ./source_wav/
   - Timbre reference: each WAV in ./reference_wav/
   - Only runs pairs where genders match:
	   * timbre gender: data/accent_archive_metadata.csv (age_sex contains male/female)
	   * style gender:  data/english_source_speakers.csv (age_sex or sex)
   - Writes:
	   output/vevostyle/timbre-<reference_stem>__style-<source_stem>.wav

This script does NOT modify anything under ./OpenVoice/ or ./Amphion/.

Run:
  python src/run_zstts.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
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
	vevostyle_out = output_root / "vevostyle"
	openvoice_out.mkdir(parents=True, exist_ok=True)
	vevostyle_out.mkdir(parents=True, exist_ok=True)
	return {
		"output_root": output_root,
		"openvoice": openvoice_out,
		"vevostyle": vevostyle_out,
	}


def orchestrate(
	*,
	openvoice_env: str,
	vevo_env: str,
	reference_wav_dir: Path,
	source_wav_dir: Path,
	reference_metadata_csv: Path,
	english_sources_csv: Path,
	output_root: Path,
	run_openvoice: bool = True,
	run_vevostyle: bool = True,
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
	# VevoStyle (gender-matched pairs only)
	# ---------
	if not run_vevostyle:
		return

	for ref_wav in inputs.reference_wavs:
		ref_g = timbre_gender.get(ref_wav.stem, "unknown")
		if ref_g == "unknown":
			print(f"[VevoStyle] skip timbre={ref_wav.stem} (unknown gender in metadata)")
			continue

		for src_wav in inputs.source_wavs:
			src_g = style_gender.get(src_wav.stem, "unknown")
			if src_g == "unknown":
				print(f"[VevoStyle] skip style={src_wav.stem} (unknown gender in english_source_speakers)")
				continue

			if src_g != ref_g:
				print(
					f"[VevoStyle] skip pair (gender mismatch): timbre={ref_wav.stem}({ref_g}) style={src_wav.stem}({src_g})"
				)
				continue

			argv = (
				_conda_python_argv(vevo_env)
				+ [
					str(self_script),
					"--mode",
					"vevostyle-child",
					"--reference-wav",
					str(ref_wav),
					"--source-wav",
					str(src_wav),
					"--output-root",
					str(dirs["output_root"]),
				]
			)
			print(f"\n[VevoStyle] timbre={ref_wav.stem}({ref_g}) style={src_wav.stem}({src_g})")
			try:
				_run(argv, cwd=repo_root)
			except subprocess.CalledProcessError as e:
				print(
					"[VevoStyle] ERROR (continuing). Pair failed: "
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


def vevostyle_child(*, reference_wav: Path, source_wav: Path, output_root: Path) -> None:
	"""Run one VevoStyle conversion (under the vevo conda env)."""

	repo_root = _repo_root()
	amphion_root = repo_root / "Amphion"
	if not amphion_root.exists():
		raise FileNotFoundError(f"Missing Amphion directory: {amphion_root}")

	reference_wav = reference_wav.expanduser().resolve()
	source_wav = source_wav.expanduser().resolve()
	if not reference_wav.exists():
		raise FileNotFoundError(f"Missing reference wav: {reference_wav}")
	if not source_wav.exists():
		raise FileNotFoundError(f"Missing source wav: {source_wav}")

	dirs = ensure_dirs(output_root.expanduser().resolve())

	# Amphion expects relative paths like ./models/... so run from Amphion root.
	os.chdir(amphion_root)
	sys.path.insert(0, str(amphion_root))

	import torch
	from huggingface_hub import snapshot_download

	from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio  # type: ignore[import-not-found]

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# Download checkpoints (if needed) into Amphion/ckpts/Vevo
	local_dir = snapshot_download(
		repo_id="amphion/Vevo",
		repo_type="model",
		cache_dir="./ckpts/Vevo",
		allow_patterns=["tokenizer/vq32/*"],
	)
	content_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl")

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

	pipeline = VevoInferencePipeline(
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

	# -----------------------------------------------------------------
	# Patch Amphion AR generation limits (no repo modifications)
	# -----------------------------------------------------------------
	# Amphion's AR model defaults to max_length=2000, but some of our VC inputs
	# exceed that length, causing Transformers to throw:
	#   ValueError: Input length ... but `max_length` is set to 2000
	# We monkeypatch the instance method to automatically increase max_length.
	orig_generate = pipeline.ar_model.generate

	def patched_generate(*args, **kwargs):
		# Support both positional and keyword calls.
		if args:
			input_ids = args[0]
			prompt_mels = args[1] if len(args) > 1 else kwargs.get("prompt_mels")
			prompt_output_ids = args[2] if len(args) > 2 else kwargs.get("prompt_output_ids")
		else:
			input_ids = kwargs.get("input_ids")
			prompt_mels = kwargs.get("prompt_mels")
			prompt_output_ids = kwargs.get("prompt_output_ids")

		max_length = int(kwargs.get("max_length", 2000))
		min_new_tokens = int(kwargs.get("min_new_tokens", 50))

		# Compute a conservative lower bound on the effective input length.
		input_len = int(getattr(input_ids, "shape", [1, 0])[1]) if input_ids is not None else 0
		prompt_len = int(getattr(prompt_output_ids, "shape", [1, 0])[1]) if prompt_output_ids is not None else 0
		# Non-global-style path concatenates input_ids + prompt_output_ids.
		effective_input_len = input_len + prompt_len
		# Global-style path also appends a style token; add a small cushion always.
		min_required = effective_input_len + max(256, min_new_tokens)
		if max_length < min_required:
			max_length = min_required

		# Call original with normalized kwargs.
		return orig_generate(
			input_ids=input_ids,
			prompt_mels=prompt_mels,
			prompt_output_ids=prompt_output_ids,
			max_length=max_length,
			temperature=kwargs.get("temperature", 0.8),
			top_k=kwargs.get("top_k", 50),
			top_p=kwargs.get("top_p", 0.9),
			repeat_penalty=kwargs.get("repeat_penalty", 1.0),
			min_new_tokens=min_new_tokens,
		)

	# Bind patched method onto this pipeline instance.
	pipeline.ar_model.generate = patched_generate

	gen_audio = pipeline.inference_ar_and_fm(
		src_wav_path=str(source_wav),
		src_text=None,
		style_ref_wav_path=str(source_wav),
		timbre_ref_wav_path=str(reference_wav),
	)

	out_wav = dirs["vevostyle"] / (
		f"timbre-{_sanitize_filename(reference_wav.stem)}__style-{_sanitize_filename(source_wav.stem)}.wav"
	)
	save_audio(gen_audio, output_path=str(out_wav))
	if not out_wav.exists() or out_wav.stat().st_size == 0:
		raise RuntimeError("[vevostyle] Inference did not create expected wav")
	print(f"[vevostyle] Wrote: {out_wav}")


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Run OpenVoice + VevoStyle (single entrypoint)")

	p.add_argument(
		"--mode",
		choices=("run", "openvoice-only", "vevostyle-only", "openvoice-child", "vevostyle-child"),
		default="run",
		help="Parent orchestrator (run) or child mode (runs inside model conda env)",
	)

	p.add_argument("--openvoice-env", default=os.environ.get("OPENVOICE_ENV", "openvoice"))
	p.add_argument("--vevo-env", default=os.environ.get("VEVO_ENV", "vevo"))

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
			vevo_env=args.vevo_env,
			reference_wav_dir=Path(args.reference_wav_dir),
			source_wav_dir=Path(args.source_wav_dir),
			reference_metadata_csv=Path(args.reference_metadata_csv),
			english_sources_csv=Path(args.english_sources_csv),
			output_root=output_root,
			run_openvoice=True,
			run_vevostyle=True,
		)
		return

	if args.mode == "openvoice-only":
		orchestrate(
			openvoice_env=args.openvoice_env,
			vevo_env=args.vevo_env,
			reference_wav_dir=Path(args.reference_wav_dir),
			source_wav_dir=Path(args.source_wav_dir),
			reference_metadata_csv=Path(args.reference_metadata_csv),
			english_sources_csv=Path(args.english_sources_csv),
			output_root=output_root,
			run_openvoice=True,
			run_vevostyle=False,
		)
		return

	if args.mode == "vevostyle-only":
		orchestrate(
			openvoice_env=args.openvoice_env,
			vevo_env=args.vevo_env,
			reference_wav_dir=Path(args.reference_wav_dir),
			source_wav_dir=Path(args.source_wav_dir),
			reference_metadata_csv=Path(args.reference_metadata_csv),
			english_sources_csv=Path(args.english_sources_csv),
			output_root=output_root,
			run_openvoice=False,
			run_vevostyle=True,
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

	if args.mode == "vevostyle-child":
		if not args.reference_wav or not args.source_wav:
			raise SystemExit("vevostyle-child requires --reference-wav and --source-wav")
		vevostyle_child(
			reference_wav=Path(args.reference_wav),
			source_wav=Path(args.source_wav),
			output_root=output_root,
		)
		return

	raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
	main()

