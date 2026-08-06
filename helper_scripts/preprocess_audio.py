from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

INPUT_OUTPUT_PAIRS = [
    ("reference", "reference_wav"),
    ("source", "source_wav"),
]

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

# Safer filter chain:
# - remove low rumble
# - remove very high-frequency noise
# - light denoise
# - normalize peak to about -3 dBFS
FILTER_CHAIN = ",".join([
    "highpass=f=50",
    "lowpass=f=7800",
    "afftdn=nf=-20",
    "alimiter=limit=0.89",
])

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install ffmpeg and make sure `ffmpeg -version` works."
        )

def iter_audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        print(f"Skipping missing folder: {folder}")
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

def build_output_path(output_dir: Path, input_file: Path) -> Path:
    return output_dir / f"{input_file.stem}.wav"

def process_file(input_file: Path, output_file: Path) -> bool:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-af", FILTER_CHAIN,
        str(output_file),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"\nFAILED: {input_file.name}")
        print(result.stderr[-2000:])
        return False

    return True

def main() -> None:
    check_ffmpeg()

    total = 0
    succeeded = 0
    failed = 0

    for input_dir_name, output_dir_name in INPUT_OUTPUT_PAIRS:
        input_dir = Path(input_dir_name)
        output_dir = Path(output_dir_name)

        files = iter_audio_files(input_dir)
        print(f"\n{input_dir} -> {output_dir}")
        print(f"Found {len(files)} files")

        for input_file in files:
            total += 1
            output_file = build_output_path(output_dir, input_file)

            print(f"Processing: {input_file} -> {output_file}")
            ok = process_file(input_file, output_file)

            if ok:
                succeeded += 1
            else:
                failed += 1

    print("\nDone.")
    print(f"Total: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()