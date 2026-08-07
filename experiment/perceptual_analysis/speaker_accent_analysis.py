from pathlib import Path

import speaker_identity_analysis as analysis


def main():
    experiment_dir = Path(__file__).resolve().parents[1]
    input_path = Path(__file__).resolve().parent / "results" / "speaker_accent_results.csv"
    output_path = experiment_dir / "figures" / "speaker_accent_model_outputs.png"
    control_path = experiment_dir / "figures" / "speaker_accent_controls.png"
    analysis.AXIS_LABEL = "Responses judged as same accent (%)"
    data = analysis.load_and_clean(input_path)
    summary = analysis.summarize(data)
    if summary[["percent", "low", "high"]].isna().any().any():
        raise ValueError("One or more expected model-tier-condition combinations are missing valid responses")
    analysis.make_output_figure(summary, output_path)
    analysis.make_control_figure(summary, control_path)
    print(summary[["label", "series", "percent", "low", "high", "responses", "participants"]].to_string(index=False))
    print(f"Saved figure to {output_path}")
    print(f"Saved figure to {control_path}")


if __name__ == "__main__":
    main()
