"""
HinEmo — Batch API Labeling, Check Status & Retrieve Results (SentiMix data)
Chunked version — handles multiple batch IDs (one per line in
data/interim/_batch_job_ids.txt) instead of a single job.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_PATH = "data/raw/twitter_takenfromsemevalpaper.csv"
BATCH_ID_FILE = "data/interim/_batch_job_ids.txt"
BATCH_OUTPUT_JSONL_TEMPLATE = "data/interim/sentimix_batch_output_part{n}.jsonl"
BATCH_ERROR_JSONL_TEMPLATE = "data/interim/sentimix_batch_errors_part{n}.jsonl"
OUTPUT_PATH = "data/processed/sentimix_batch_labeled.csv"

VALID_LABELS = {"anger", "disgust", "fear", "sadness", "joy", "surprise", "neutral"}


def load_batch_ids():
    if not os.path.exists(BATCH_ID_FILE):
        raise FileNotFoundError(
            f"No batch ID file found at {BATCH_ID_FILE}. "
            "Run scripts/6a_batch_submit_job.py first."
        )
    with open(BATCH_ID_FILE) as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"{BATCH_ID_FILE} is empty.")
    return ids


def download_file(file_id, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    content = client.files.content(file_id)
    with open(out_path, "wb") as f:
        f.write(content.content)


def parse_output_jsonl(path, results):
    """Adds custom_id -> label entries from one chunk's output file into
    the shared `results` dict."""
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
            if label not in VALID_LABELS:
                print(f"[!] Unexpected label for custom_id={custom_id}: {content!r}")
            results[custom_id] = label


def main():
    batch_ids = load_batch_ids()
    print(f"Tracking {len(batch_ids)} batch job(s)\n")

    all_completed = True
    results = {}

    for i, batch_id in enumerate(batch_ids, start=1):
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        status_line = f"[{i}/{len(batch_ids)}] {batch.id}: {batch.status}"
        if counts is not None:
            status_line += f" ({counts.completed}/{counts.total} completed, {counts.failed} failed)"
        print(status_line)

        if batch.status in ("validating", "in_progress", "finalizing"):
            all_completed = False
            continue

        if batch.status in ("failed", "expired", "cancelled"):
            all_completed = False
            print(f"  [!] Chunk {i} ended with status '{batch.status}'")
            if batch.error_file_id:
                err_path = BATCH_ERROR_JSONL_TEMPLATE.format(n=i)
                download_file(batch.error_file_id, err_path)
                print(f"  Saved error details to {err_path}")
            continue

        if batch.status == "completed" and batch.output_file_id:
            out_path = BATCH_OUTPUT_JSONL_TEMPLATE.format(n=i)
            download_file(batch.output_file_id, out_path)
            parse_output_jsonl(out_path, results)
            print(f"  Downloaded and parsed results from {out_path}")

    if not all_completed:
        print("\nNot all chunks are finished/successful yet. Run this script again later.")
        print("(Any chunks that already completed have had their results saved.)")
        return

    print(f"\nAll {len(batch_ids)} chunk(s) completed. Parsed {len(results)} labeled results total.")

    df = pd.read_csv(INPUT_PATH)
    df["source_id"] = df["source_id"].astype(str)
    if df["source_id"].duplicated().any():
        df = df.drop_duplicates(subset=["source_id"])

    df["emotion"] = df["source_id"].map(results)

    n_missing = df["emotion"].isna().sum()
    if n_missing:
        print(f"[!] {n_missing} rows have no matching label (failed request or missing from batch).")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved labeled data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()