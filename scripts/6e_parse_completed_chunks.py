"""
HinEmo — Parse Already-Downloaded Batch Output Files
===========================================================
Standalone script — no API calls, no waiting. Just parses whichever
sentimix_batch_output_part{n}.jsonl files already exist locally and merges
them into a labeled CSV. Safe to rerun anytime as more chunks complete and
get downloaded — it picks up whatever part files are present.
"""

import os
import glob
import json
import pandas as pd

INPUT_PATH = "data/raw/twitter_takenfromsemevalpaper.csv"
OUTPUT_JSONL_PATTERN = "data/interim/sentimix_batch_output_part*.jsonl"
OUTPUT_PATH = "data/processed/sentimix_batch_labeled.csv"

VALID_LABELS = {"anger", "disgust", "fear", "sadness", "joy", "surprise", "neutral"}


def parse_output_jsonl(path, results):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            response = record.get("response")

            if response is None or response.get("status_code") != 200:
                results[custom_id] = None
                continue

            try:
                content = response["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                results[custom_id] = None
                continue

            label = content.strip().lower().strip(" .'\"")
            results[custom_id] = label if label in VALID_LABELS else None


def main():
    part_files = sorted(glob.glob(OUTPUT_JSONL_PATTERN))
    if not part_files:
        print(f"No output files found matching {OUTPUT_JSONL_PATTERN}")
        return

    print(f"Found {len(part_files)} output file(s):")
    for p in part_files:
        print(f"  {p}")

    results = {}
    for path in part_files:
        parse_output_jsonl(path, results)

    print(f"\nParsed {len(results)} total labeled results")

    df = pd.read_csv(INPUT_PATH)
    df["source_id"] = df["source_id"].astype(str)
    if df["source_id"].duplicated().any():
        df = df.drop_duplicates(subset=["source_id"])

    df["emotion"] = df["source_id"].map(results)

    matched = df["emotion"].notna().sum()
    print(f"Matched {matched}/{len(df)} rows in the source data")

    df_labeled = df[df["emotion"].notna()].copy()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_labeled.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_labeled)} labeled rows -> {OUTPUT_PATH}")
    print("\nLabel distribution so far:")
    print(df_labeled["emotion"].value_counts())


if __name__ == "__main__":
    main()