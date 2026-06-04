## Accent Conversion

This project evaluates zero-shot **accent conversion** — transferring the timbre and accent of a non-native English speaker onto American English source audio — using two voice conversion models: **OpenVoice v2** and **SeedVC**.

The goal is to understand how well each model preserves the target speaker's voice identity (timbre) while using the content of a different source utterance, with a focus on speakers from Hindi- and Gujarati-accented English backgrounds.

---

### Pipeline

```
raw audio
    └─ preprocess_audio.py    16 kHz mono WAV, ffmpeg filter chain (highpass/lowpass/denoise/normalize)
           └─ run_zstts.py    Gender-matched inference: OpenVoice v2 + SeedVC
                  └─ run_mfa.py     Phoneme-level forced alignment via Montreal Forced Aligner
                         └─ textgrid_cleaning.py   Clean and standardize TextGrids
                                └─ timbre_similarity.py   Cosine similarity + mixed-effects regression
```

**Inputs:**
- `reference_wav/` — accent-archive speakers (Hindi/Gujarati L1, English L2); gender from `data/accent_archive_metadata.csv`
- `source_wav/` — American English speakers; gender from `data/english_source_speakers.csv`
- Only gender-matched reference–source pairs are run through inference

**Outputs:**
- `output/openvoice/` — converted audio from OpenVoice v2
- `output/seed_vc/` — converted audio from SeedVC
- `*_textgrids_cleaned/` — forced-aligned and cleaned TextGrids for all audio
- `analysis_files/` — ranked audio samples, figures, top/bottom-5 pairs per model

---

### Initial Findings: Timbre Similarity

Speaker similarity between each timbre reference and its model output was measured using [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) d-vector embeddings (256-dim utterance-level cosine similarity). 51 reference–output pairs were evaluated per model.

| Model     | Mean ± SD         | Min    | Max    |
|-----------|-------------------|--------|--------|
| OpenVoice | 0.8664 ± 0.0392   | 0.7335 | 0.9450 |
| SeedVC    | 0.9633 ± 0.0119   | 0.9199 | 0.9819 |

A linear mixed-effects model (`similarity ~ model_type + (1 | speaker)`) was fit to test whether model type predicts similarity while controlling for speaker-level variability:

- SeedVC produced cosine similarities **+0.097 higher** than OpenVoice on average (95% CI [0.086, 0.108], p < 0.001)
- The speaker random-intercept variance was essentially zero (0.000073), meaning the difficulty of a pair is driven by the model, not the speaker identity

**Per-pair rankings:**

| | Top 5 | Bottom 5 |
|---|---|---|
| OpenVoice | hindi19, gujarati17, gujarati11, hindi13, gujarati13 | gujarati9, gujarati8, hindi7, hindi12, gujarati7 |
| SeedVC | gujarati5, gujarati16, hindi19, hindi2, hindi10 | hindi21, hindi16, hindi26, hindi13, gujarati4 |

Full rankings and per-pair grouped bar charts are in [`result_descriptions/timbre_similarity_results.md`](result_descriptions/timbre_similarity_results.md).

---

### Structure

```
src/
  preprocess_audio.py     Audio normalization and resampling
  run_zstts.py            Orchestrates OpenVoice + SeedVC inference (gender-matched pairs)
  run_mfa.py              Incremental Montreal Forced Aligner wrapper
  add_transcripts.py      Attaches transcripts to audio for MFA
  textgrid_cleaning.py    Cleans/standardizes MFA TextGrid output
  timbre_similarity.py    Resemblyzer similarity scoring, regression, figures

OpenVoice/               OpenVoice v2 (submodule)
seed-vc/                 SeedVC (submodule)
data/                    Speaker metadata CSVs
source/, reference/      Raw audio (pre-processing)
source_wav/, reference_wav/  Preprocessed audio
output/                  Model outputs
analysis_files/          Top/bottom pairs and figures
result_descriptions/     Written results summaries
```
