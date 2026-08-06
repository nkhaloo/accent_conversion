from pathlib import Path


TRANSCRIPT = (
    "Please call Stella. Ask her to bring these things with her from the store: "
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack "
    "for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. "
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the "
    "train station."
)


def write_lab_files(corpus_dir: Path, transcript: str, *, overwrite: bool = False) -> None:
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    for wav in corpus_dir.glob("*.wav"):
        lab = wav.with_suffix(".lab")
        if lab.exists() and not overwrite:
            continue
        lab.write_text(transcript.strip() + "\n", encoding="utf8")


for folder in ("reference_wav", "source_wav"):
    write_lab_files(Path(folder), TRANSCRIPT, overwrite=False)