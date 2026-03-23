import os
import subprocess
from pathlib import Path


MFA_ENV_BIN = Path("/opt/anaconda3/envs/accent_conversion/bin")
MFA_BIN = str(MFA_ENV_BIN / "mfa")


def run_align(corpus_dir: Path, output_dir: Path) -> None:
    corpus_dir = corpus_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[MFA] Aligning corpus: {corpus_dir}")
    print(f"[MFA] Output TextGrids: {output_dir}\n")

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


run_align(Path("reference_wav"), Path("reference_textgrids"))
run_align(Path("source_wav"), Path("source_textgrids"))