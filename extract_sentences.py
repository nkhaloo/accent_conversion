"""
Extract sentence 1 ("Please call Stella.") and sentence 2
("Ask her to bring these things with her from the store")
for every top5/bottom5 listening-test item: the converted output,
its timbre reference, and its source content audio.

Boundaries come from each file's own TextGrid:
  interval[11] = silence after "Stella"  → s1 ends at its xmax
  interval[12] = start of "Ask"          → s2 starts at its xmin
  interval[39] = silence after "store"   → s2 ends at its xmax

top5/bottom5 membership is read directly from analysis_files/{model}/{top5,bottom5}/,
which src/timbre_similarity.py populates -- there is no separate hardcoded list here,
so this can never drift out of sync with the actual rankings.

Output layout (identical for every audio kind):
  analysis_files/sentence_extracts/{model}/{label}/{kind}/sentence{1,2}/{basename}_s{1,2}.wav
  analysis_files/sentence_extracts/{model}/{label}/{kind}/sentence{1,2}/{basename}_s{1,2}.TextGrid

where model in {openvoice, seed_vc}, label in {top5, bottom5},
kind in {output, reference, source}. Reference/source clips are copied once per
(model, label) pair that needs them, so a speaker appearing in different labels
across models (e.g. top5 for one model, bottom5 for the other) gets both copies.
"""

import os
import re
import subprocess

BASE = "/Users/noahkhaloo/Downloads/projects/accent_conversion"

MODELS = ["openvoice", "seed_vc"]
LABELS = ["top5", "bottom5"]

RANKED_DIR = os.path.join(BASE, "analysis_files")

OUTPUT_TG_DIR = {
    "openvoice": os.path.join(BASE, "output", "openvoice", "textgrids_cleaned"),
    "seed_vc":   os.path.join(BASE, "output", "seed_vc",   "textgrids_cleaned"),
}

SOURCE_DIR    = os.path.join(BASE, "source")
SOURCE_TG_DIR = os.path.join(BASE, "source_textgrids_cleaned")

REF_TG_DIR = os.path.join(BASE, "reference_textgrids_cleaned")

OUT_BASE = os.path.join(BASE, "analysis_files", "sentence_extracts")

TIMBRE_RE = re.compile(r"^timbre-(.+?)__source-")
SOURCE_ID_RE = re.compile(r"__source-(english\d+__\d+)__")


def parse_intervals(tg_path):
    with open(tg_path) as f:
        content = f.read()
    pattern = r'intervals \[(\d+)\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"([^\"]*)\"'
    rows = re.findall(pattern, content)
    return [(int(i), float(xmin), float(xmax), text) for i, xmin, xmax, text in rows]


TAIL_PAD = 0.08  # seconds of silence to keep after last phone

def get_sentence_times(tg_path):
    ivs = parse_intervals(tg_path)
    # 0-indexed: interval[11] → ivs[10], interval[12] → ivs[11]
    # interval[38] → ivs[37]  (last phone "R" of "store")
    # interval[39] → ivs[38]  (silence after store — NOT used as end point)
    s1_end   = ivs[10][2]          # xmax of interval[11]  (silence after Stella)
    s2_start = ivs[11][1]          # xmin of interval[12]  (start of Ask)
    s2_end   = ivs[37][2] + TAIL_PAD  # end of "R" in "store" + small tail
    return s1_end, s2_start, s2_end


def ffmpeg_extract(src, start, end, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ss", str(start), "-to", str(end), dst],
        check=True, capture_output=True,
    )


def write_textgrid_excerpt(intervals, start, end, dst):
    """Write a TextGrid containing only the intervals that fall within [start, end],
    time-shifted so the excerpt begins at 0."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    duration = end - start

    clipped = []
    for _, xmin, xmax, text in intervals:
        if xmax <= start or xmin >= end:
            continue
        new_xmin = max(xmin - start, 0.0)
        new_xmax = min(xmax - start, duration)
        clipped.append((new_xmin, new_xmax, text))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0 ",
        f"xmax = {duration} ",
        "tiers? <exists> ",
        "size = 1 ",
        "item []: ",
        "    item [1]:",
        '        class = "IntervalTier" ',
        '        name = "phones" ',
        "        xmin = 0 ",
        f"        xmax = {duration} ",
        f"        intervals: size = {len(clipped)} ",
    ]
    for idx, (xmin, xmax, text) in enumerate(clipped, 1):
        lines += [
            f"        intervals [{idx}]:",
            f"            xmin = {xmin} ",
            f"            xmax = {xmax} ",
            f'            text = "{text}" ',
        ]

    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n")


def extract_sentence_pair(tg_path, wav_path, out_dir, basename):
    """Shared extraction logic for every audio kind (output/reference/source):
    locate the two sentences in tg_path/wav_path and write trimmed wav+TextGrid
    pairs into out_dir/sentence{1,2}/."""
    if not os.path.exists(tg_path):
        print(f"  WARNING: TextGrid not found: {tg_path}")
        return False
    if not os.path.exists(wav_path):
        print(f"  WARNING: audio not found: {wav_path}")
        return False

    s1_end, s2_start, s2_end = get_sentence_times(tg_path)
    intervals = parse_intervals(tg_path)

    out_s1_wav = os.path.join(out_dir, "sentence1", f"{basename}_s1.wav")
    out_s2_wav = os.path.join(out_dir, "sentence2", f"{basename}_s2.wav")
    out_s1_tg  = os.path.join(out_dir, "sentence1", f"{basename}_s1.TextGrid")
    out_s2_tg  = os.path.join(out_dir, "sentence2", f"{basename}_s2.TextGrid")

    ffmpeg_extract(wav_path, 0.0, s1_end, out_s1_wav)
    ffmpeg_extract(wav_path, s2_start, s2_end, out_s2_wav)
    write_textgrid_excerpt(intervals, 0.0,      s1_end, out_s1_tg)
    write_textgrid_excerpt(intervals, s2_start, s2_end, out_s2_tg)
    print(f"    {basename}: s1=0–{s1_end:.2f}s  s2={s2_start:.2f}–{s2_end:.2f}s")
    return True


def discover_labels(model):
    """Read analysis_files/{model}/{top5,bottom5}/ (written by timbre_similarity.py)
    to find which reference ids belong to each label -- this is the single source
    of truth for top5/bottom5 membership, so it can't drift from the actual rankings."""
    labels = {}
    for label in LABELS:
        label_dir = os.path.join(RANKED_DIR, model, label)
        ref_ids = []
        if os.path.isdir(label_dir):
            for fname in sorted(os.listdir(label_dir)):
                m = TIMBRE_RE.match(fname)
                if m:
                    ref_ids.append(m.group(1))
        else:
            print(f"  WARNING: {label_dir} not found")
        labels[label] = ref_ids
    return labels


def find_output_tg(tg_dir, ref_id):
    matches = [f for f in os.listdir(tg_dir)
               if f.startswith(f"timbre-{ref_id}__source-") and f.endswith(".TextGrid")]
    return matches[0] if matches else None


def main():
    for model in MODELS:
        tg_dir = OUTPUT_TG_DIR[model]
        labels = discover_labels(model)
        print(f"\n=== {model} ===")

        for label, ref_ids in labels.items():
            print(f"  -- {label} ({len(ref_ids)} speakers) --")

            for ref_id in ref_ids:
                tg_file = find_output_tg(tg_dir, ref_id)
                if not tg_file:
                    print(f"  WARNING: no output TextGrid found for {model}/{label}/{ref_id}")
                    continue

                basename = tg_file.replace(".TextGrid", "")
                wav_file = tg_file.replace(".TextGrid", ".wav")

                extract_sentence_pair(
                    os.path.join(tg_dir, tg_file),
                    os.path.join(tg_dir, wav_file),
                    os.path.join(OUT_BASE, model, label, "output"),
                    basename,
                )

                extract_sentence_pair(
                    os.path.join(REF_TG_DIR, f"{ref_id}.TextGrid"),
                    os.path.join(REF_TG_DIR, f"{ref_id}.wav"),
                    os.path.join(OUT_BASE, model, label, "reference"),
                    ref_id,
                )

                m = SOURCE_ID_RE.search(tg_file)
                if not m:
                    print(f"  WARNING: could not parse source id from {tg_file}")
                    continue
                source_id = m.group(1)

                extract_sentence_pair(
                    os.path.join(SOURCE_TG_DIR, f"{source_id}.TextGrid"),
                    os.path.join(SOURCE_DIR, f"{source_id}.mp3"),
                    os.path.join(OUT_BASE, model, label, "source"),
                    source_id,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
