"""
HinEmo Data Collection — SentiMix Base Loader
================================================
Loads the SemEval-2020 Task 9 SentiMix Hinglish corpus (Patwa et al. 2020) —
the same corpus Paper 1 used. It ships real tweet text in CoNLL format with
GOLD word-level language tags (Hin/Eng/O) already assigned. This gives you
~20,000 tweets for free, with the hardest part of Step 1.3 (language tagging)
already done for this portion of the data.

If you still have the train/val/test .conll or .txt files from Paper 1,
point SENTIMIX_DIR at that folder and skip re-downloading.

CoNLL format (one token per line, blank line between tweets):
    meta	<tweet_id>	<sentiment>
    token1	<lang_tag>
    token2	<lang_tag>
    ...

This script:
  1. Parses the CoNLL files into (tweet_id, full_text, token_lang_tags, sentiment)
  2. Reconstructs the surface text (joining tokens)
  3. Saves a clean CSV ready for Step 3 (GPT emotion labeling)

NOTE: this only recovers SENTIMENT labels + language tags, not emotion labels.
Emotion labels get added in 3_label_emotions_gpt.py, same as everything else.
"""

import os
import re
import pandas as pd

SENTIMIX_DIR = "./sentimix_raw"  # point this at your Paper 1 SentiMix files
OUTPUT_PATH = "sentimix_base_with_lang_tags.csv"


def parse_conll_file(filepath):
    """Parses one SentiMix CoNLL-format file into a list of tweet records."""
    records = []
    current_id, current_sentiment = None, None
    tokens, lang_tags = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    records.append({
                        "tweet_id": current_id,
                        "sentiment": current_sentiment,
                        "text": " ".join(tokens),
                        "tokens": tokens.copy(),
                        "lang_tags": lang_tags.copy(),
                    })
                tokens, lang_tags = [], []
                continue

            if line.startswith("meta"):
                parts = line.split("\t")
                # meta \t tweet_id \t sentiment
                current_id = parts[1] if len(parts) > 1 else None
                current_sentiment = parts[2] if len(parts) > 2 else None
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    lang_tags.append(parts[1])

        # flush last tweet if file doesn't end with blank line
        if tokens:
            records.append({
                "tweet_id": current_id,
                "sentiment": current_sentiment,
                "text": " ".join(tokens),
                "tokens": tokens.copy(),
                "lang_tags": lang_tags.copy(),
            })
    return records


def main():
    if not os.path.isdir(SENTIMIX_DIR):
        print(f"[!] {SENTIMIX_DIR} not found.")
        print("    Point SENTIMIX_DIR at the folder containing your Paper 1")
        print("    SentiMix train/val/test CoNLL files, then rerun.")
        print("    If you no longer have them, the corpus is the SemEval-2020")
        print("    Task 9 'SentiMix Hindi-English' release (Patwa et al. 2020).")
        return

    all_records = []
    for fname in os.listdir(SENTIMIX_DIR):
        if fname.endswith((".txt", ".conll", ".tsv")):
            fpath = os.path.join(SENTIMIX_DIR, fname)
            recs = parse_conll_file(fpath)
            print(f"  parsed {len(recs)} tweets from {fname}")
            all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["tweet_id"])
    df["source"] = "sentimix"
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} tweets with gold language tags to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
