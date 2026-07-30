"""
HinEmo — Download & Parse teaser5k Batch Results
======================================================
Same pattern as 6f + 6e for SentiMix, adapted for teaser5k's id/text columns.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

INPUT_PATH = "data/Extra(fortesting)/hinglish_teaser_5k.json"
BATCH_ID_FILE = "data/interim/_batch_job_ids_teaser5k.txt"
OUTPUT_JSONL_TEMPLATE = "data/interim/teaser5k_batch_output_{suffix}.jsonl"
FINAL_OUTPUT_PATH = "data/processed/teaser5k_labeled.csv"

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
                continue
            try:
                content = response["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                continue
            label = content.strip().lower().strip(" .'\"")
            if label in VALID_LABELS:
                results[custom_id] = label


def main():
    with open(BATCH_ID_FILE) as f:
        batch_ids = [line.strip() for line in f if line.strip()]

    results = {}
    all_done = True

    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        print(f"{batch_id}: {batch.status}")

        if batch.status != "completed":
            all_done = False
            continue

        if batch.output_file_id:
            out_path = OUTPUT_JSONL_TEMPLATE.format(suffix=batch_id[-8:])
            if not os.path.exists(out_path):
                content = client.files.content(batch.output_file_id)
                with open(out_path, "wb") as f:
                    f.write(content.content)
            parse_output_jsonl(out_path, results)

    print(f"\nParsed {len(results)} labeled results so far")
    if not all_done:
        print("Not all chunks completed yet — rerun later for the rest.")

    with open(INPUT_PATH) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["id"] = df["id"].astype(str)
    df["emotion"] = df["id"].map(results)

    df_labeled = df[df["emotion"].notna()].copy()
    os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)
    df_labeled.to_csv(FINAL_OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df_labeled)} labeled rows -> {FINAL_OUTPUT_PATH}")
    print(df_labeled["emotion"].value_counts())


if __name__ == "__main__":
    main()