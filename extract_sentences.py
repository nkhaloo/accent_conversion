"""
Extract sentence 1 ("Please call Stella.") and sentence 2
("Ask her to bring these things with her from the store")
for top-5 and bottom-5 model outputs plus their source audio.

Boundaries come from each file's own TextGrid:
  interval[11] = silence after "Stella"  → s1 ends at its xmax
  interval[12] = start of "Ask"          → s2 starts at its xmin
  interval[39] = silence after "store"   → s2 ends at its xmax
"""

import os
import re
import subprocess

BASE = "/Users/noahkhaloo/Downloads/projects/accent_conversion"

MODELS = {
    "openvoice": {
        "top5":    ["hindi19", "gujarati17", "gujarati11", "hindi13", "gujarati13"],
        "bottom5": ["gujarati9", "gujarati8", "hindi7", "hindi12", "gujarati7"],
    },
    "seed_vc": {
        "top5":    ["gujarati5", "gujarati16", "hindi19", "hindi2", "hindi10"],
        "bottom5": ["hindi21", "hindi16", "hindi26", "hindi13", "gujarati4"],
    },
}

OUTPUT_TG_DIR = {
    "openvoice": os.path.join(BASE, "output", "openvoice", "textgrids_cleaned"),
    "seed_vc":   os.path.join(BASE, "output", "seed_vc",   "textgrids_cleaned"),
}

SOURCE_DIR    = os.path.join(BASE, "source")
SOURCE_TG_DIR = os.path.join(BASE, "source_textgrids_cleaned")

REF_TG_DIR = os.path.join(BASE, "reference_textgrids_cleaned")

OUT_BASE = os.path.join(BASE, "analysis_files", "sentence_extracts")


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


def process_output_file(model, label, speaker, tg_dir):
    # Locate the TextGrid and WAV
    tg_files = [f for f in os.listdir(tg_dir)
                if f.startswith(f"timbre-{speaker}__") and f.endswith(".TextGrid")]
    if not tg_files:
        print(f"  WARNING: no TextGrid found for {model}/{speaker}")
        return
    tg_file = tg_files[0]
    wav_file = tg_file.replace(".TextGrid", ".wav")

    tg_path  = os.path.join(tg_dir, tg_file)
    wav_path = os.path.join(tg_dir, wav_file)

    if not os.path.exists(wav_path):
        print(f"  WARNING: WAV not found: {wav_path}")
        return

    s1_end, s2_start, s2_end = get_sentence_times(tg_path)
    intervals = parse_intervals(tg_path)

    base = tg_file.replace(".TextGrid", "")
    out_s1_wav = os.path.join(OUT_BASE, model, label, "sentence1", f"{base}_s1.wav")
    out_s2_wav = os.path.join(OUT_BASE, model, label, "sentence2", f"{base}_s2.wav")
    out_s1_tg  = os.path.join(OUT_BASE, model, label, "sentence1", f"{base}_s1.TextGrid")
    out_s2_tg  = os.path.join(OUT_BASE, model, label, "sentence2", f"{base}_s2.TextGrid")

    ffmpeg_extract(wav_path, 0.0, s1_end, out_s1_wav)
    ffmpeg_extract(wav_path, s2_start, s2_end, out_s2_wav)
    write_textgrid_excerpt(intervals, 0.0,      s1_end, out_s1_tg)
    write_textgrid_excerpt(intervals, s2_start, s2_end, out_s2_tg)
    print(f"  {speaker}: s1=0–{s1_end:.2f}s  s2={s2_start:.2f}–{s2_end:.2f}s")


def process_source_file(source_id):
    tg_path  = os.path.join(SOURCE_TG_DIR, f"{source_id}.TextGrid")
    mp3_path = os.path.join(SOURCE_DIR, f"{source_id}.mp3")

    if not os.path.exists(mp3_path):
        print(f"  WARNING: source MP3 not found: {mp3_path}")
        return

    s1_end, s2_start, s2_end = get_sentence_times(tg_path)
    intervals = parse_intervals(tg_path)

    out_s1_wav = os.path.join(OUT_BASE, "source", "sentence1", f"{source_id}_s1.wav")
    out_s2_wav = os.path.join(OUT_BASE, "source", "sentence2", f"{source_id}_s2.wav")
    out_s1_tg  = os.path.join(OUT_BASE, "source", "sentence1", f"{source_id}_s1.TextGrid")
    out_s2_tg  = os.path.join(OUT_BASE, "source", "sentence2", f"{source_id}_s2.TextGrid")

    ffmpeg_extract(mp3_path, 0.0, s1_end, out_s1_wav)
    ffmpeg_extract(mp3_path, s2_start, s2_end, out_s2_wav)
    write_textgrid_excerpt(intervals, 0.0,      s1_end, out_s1_tg)
    write_textgrid_excerpt(intervals, s2_start, s2_end, out_s2_tg)
    print(f"  source/{source_id}: s1=0–{s1_end:.2f}s  s2={s2_start:.2f}–{s2_end:.2f}s")


def process_reference_file(full_speaker_id, label):
    tg_path  = os.path.join(REF_TG_DIR, f"{full_speaker_id}.TextGrid")
    wav_path = os.path.join(REF_TG_DIR, f"{full_speaker_id}.wav")

    if not os.path.exists(tg_path):
        print(f"  WARNING: reference TextGrid not found: {tg_path}")
        return
    if not os.path.exists(wav_path):
        print(f"  WARNING: reference WAV not found: {wav_path}")
        return

    s1_end, s2_start, s2_end = get_sentence_times(tg_path)
    intervals = parse_intervals(tg_path)

    out_s1_wav = os.path.join(OUT_BASE, "reference", label, "sentence1", f"{full_speaker_id}_s1.wav")
    out_s2_wav = os.path.join(OUT_BASE, "reference", label, "sentence2", f"{full_speaker_id}_s2.wav")
    out_s1_tg  = os.path.join(OUT_BASE, "reference", label, "sentence1", f"{full_speaker_id}_s1.TextGrid")
    out_s2_tg  = os.path.join(OUT_BASE, "reference", label, "sentence2", f"{full_speaker_id}_s2.TextGrid")

    ffmpeg_extract(wav_path, 0.0, s1_end, out_s1_wav)
    ffmpeg_extract(wav_path, s2_start, s2_end, out_s2_wav)
    write_textgrid_excerpt(intervals, 0.0,      s1_end, out_s1_tg)
    write_textgrid_excerpt(intervals, s2_start, s2_end, out_s2_tg)
    print(f"  {full_speaker_id}: s1=0–{s1_end:.2f}s  s2={s2_start:.2f}–{s2_end:.2f}s")


def main():
    sources_needed = set()
    refs_needed = {}  # full_speaker_id → label

    for model, groups in MODELS.items():
        tg_dir = OUTPUT_TG_DIR[model]
        print(f"\n=== {model} ===")
        for label, speakers in groups.items():
            print(f"  -- {label} --")
            for speaker in speakers:
                process_output_file(model, label, speaker, tg_dir)
                tg_files = [f for f in os.listdir(tg_dir)
                            if f.startswith(f"timbre-{speaker}__") and f.endswith(".TextGrid")]
                if tg_files:
                    m = re.search(r"__source-(english\d+__\d+)__", tg_files[0])
                    if m:
                        sources_needed.add(m.group(1))
                    m2 = re.match(r"timbre-(\w+__\d+)__", tg_files[0])
                    if m2:
                        refs_needed[m2.group(1)] = label

    print("\n=== source ===")
    for src_id in sorted(sources_needed):
        process_source_file(src_id)

    print("\n=== reference ===")
    for ref_id, label in sorted(refs_needed.items()):
        process_reference_file(ref_id, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
