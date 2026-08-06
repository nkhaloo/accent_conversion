from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from praatio import textgrid

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}

SONORANTS = {
    "M", "N", "NG",
    "L", "R",
    "W", "Y",
}

_STRESS_DIGITS = re.compile(r"\d+")


def normalize_phone(label: str) -> str:
    """
    Convert MFA ARPAbet phones like 'AA1' or 'IH0' -> 'AA'/'IH'.
    """
    return _STRESS_DIGITS.sub("", label.strip().upper())



def is_vowel_or_sonorant(label: str) -> bool:
    p = normalize_phone(label)
    return (p in VOWELS) or (p in SONORANTS)



def _get_interval_tier(tg: textgrid.Textgrid, tier_index: int) -> textgrid.IntervalTier:
    tiers = list(tg.tiers)
    if tier_index < 0 or tier_index >= len(tiers):
        raise ValueError(f"TextGrid has {len(tiers)} tiers, can't access index {tier_index}.")
    tier = tiers[tier_index]
    if not isinstance(tier, textgrid.IntervalTier):
        raise TypeError(f"Tier {tier_index} is not an IntervalTier: {type(tier)}")
    return tier


def filter_textgrid_keep_vowels_sonorants(
    tg_path: Path,
    out_path: Path,
    *,
    keep_from_tier_index: int = 1,
) -> None:
    """
    - Reads a TextGrid
    - Discards tier 1 entirely
    - Creates a new TextGrid with only 1 tier:
        tier 2 filtered to keep only vowels+sonorants
    """
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

    source_tier = _get_interval_tier(tg, keep_from_tier_index)

    kept_entries = []
    for interval in source_tier.entries:
        label = interval.label or ""
        if label.strip() and is_vowel_or_sonorant(label):
            kept_entries.append(interval)

    new_tg = textgrid.Textgrid()
    new_tg.addTier(
        textgrid.IntervalTier(
            name=source_tier.name,
            entries=kept_entries,
            minT=tg.minTimestamp,
            maxT=tg.maxTimestamp,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_tg.save(str(out_path), format="long_textgrid", includeBlankSpaces=True)



def process_textgrid_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    keep_from_tier_index: int = 1,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tg_paths = sorted(input_dir.glob("*.TextGrid"))
    if not tg_paths:
        raise FileNotFoundError(f"No .TextGrid files found in: {input_dir}")

    for tg_path in tg_paths:
        out_path = output_dir / tg_path.name
        filter_textgrid_keep_vowels_sonorants(
            tg_path,
            out_path,
            keep_from_tier_index=keep_from_tier_index,
        )

    print(f"Processed {len(tg_paths)} TextGrids -> {output_dir}")


if __name__ == "__main__":
    process_textgrid_folder(
        Path("output/openvoice/textgrids"),
        Path("output/openvoice/textgrids_cleaned"),
        keep_from_tier_index=1,
    )

    process_textgrid_folder(
        Path("output/seed_vc/textgrids"),
        Path("output/seed_vc/textgrids_cleaned"),
        keep_from_tier_index=1,
    )