"""
HinEmo — Clean Targeted Round 2 Data (steps 1-5, pre-IndicLID)
=====================================================================
Applies the exact same local cleaning rules used on the original dataset,
in the same order. Output feeds into the existing IndicLID Colab notebook
for the language-bucketing step (can't run locally — needs the downloaded
model weights).

Input:  data/raw/youtube_targeted_round2.csv
Output: data/interim/round2_cleaned_stage2.csv (ready for Colab/IndicLID)
"""

import re
import string
import pandas as pd

INPUT_PATH = "data/raw/youtube_targeted_round2.csv"
OUTPUT_PATH = "data/interim/round2_cleaned_stage2.csv"

MIN_TOKENS = 4

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")

OTHER_SCRIPT_RANGES = {
    "gurmukhi_punjabi": (0x0A00, 0x0A7F),
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
}


def basic_clean(text):
    text = URL_RE.sub("", str(text))
    text = MENTION_RE.sub("", text)
    return text.strip()


def token_count(text):
    return len(text.split())


def detect_other_scripts(text):
    found = set()
    for char in str(text):
        cp = ord(char)
        for script, (start, end) in OTHER_SCRIPT_RANGES.items():
            if start <= cp <= end:
                found.add(script)
    return found


def has_latin_letters(text):
    return any(c in string.ascii_letters for c in str(text))


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Step 1: basic clean
    df["text_clean"] = df["text"].apply(basic_clean)

    # Step 2: dedup by source_id
    before = len(df)
    df = df.drop_duplicates(subset=["source_id"])
    print(f"After dedup: {len(df)} (-{before - len(df)})")

    # Step 3: min length filter
    before = len(df)
    df = df[df["text_clean"].apply(token_count) >= MIN_TOKENS]
    print(f"After min-length filter (>= {MIN_TOKENS} tokens): {len(df)} (-{before - len(df)})")

    # Step 4: drop other Indic scripts
    df["other_scripts"] = df["text_clean"].apply(detect_other_scripts)
    df["has_other_script"] = df["other_scripts"].apply(lambda s: len(s) > 0)
    before = len(df)
    df = df[~df["has_other_script"]].drop(columns=["other_scripts", "has_other_script"])
    print(f"After dropping other-script rows: {len(df)} (-{before - len(df)})")

    # Step 5: drop pure-Devanagari (no Latin characters)
    df["has_latin"] = df["text_clean"].apply(has_latin_letters)
    before = len(df)
    df = df[df["has_latin"]].drop(columns=["has_latin"])
    print(f"After dropping pure-Devanagari: {len(df)} (-{before - len(df)})")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows -> {OUTPUT_PATH}")
    print("Next: upload this file to the Colab notebook for IndicLID language")
    print("bucketing (steps 6-8), same process as the original dataset.")


if __name__ == "__main__":
    main()