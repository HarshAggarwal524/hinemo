"""
HinEmo — Download round-2 batch outputs and merge labels into the CSV.
"""

import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID_FILE = "data/interim/_batch_job_ids_round2.txt"
SOURCE_CSV = "data/interim/round2_mixed_stage3.csv"
OUTPUT_CSV = "data/interim/round2_labeled.csv"
RAW_OUTPUT_DIR = "data/interim/round2_batch_output_part{n}.jsonl"

VALID_LABELS = {"anger", "disgust", "fear", "sadness", "joy", "surprise", "neutral"}


def main():
    with open(BATCH_ID_FILE) as f:
        batch_ids = [line.strip() for line in f if line.strip()]

    labels = {}  # source_id -> label
    malformed = []  # (source_id, raw_response)
    request_errors = []  # (source_id, error) from error_file_id

    for i, bid in enumerate(batch_ids, start=1):
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            print(f"Chunk {i} ({bid}): status is '{b.status}', not 'completed' — skipping")
            continue

        # --- successful outputs ---
        if b.output_file_id:
            content = client.files.content(b.output_file_id).text
            raw_path = RAW_OUTPUT_DIR.format(n=i)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(content)

            for line in content.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record["custom_id"]
                body = record.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    malformed.append((custom_id, json.dumps(record)[:200]))
                    continue

                raw_label = choices[0]["message"]["content"].strip().lower()
                if raw_label in VALID_LABELS:
                    labels[custom_id] = raw_label
                else:
                    malformed.append((custom_id, raw_label))

        # --- per-request errors within an otherwise-completed batch ---
        if b.error_file_id:
            err_content = client.files.content(b.error_file_id).text
            for line in err_content.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                request_errors.append((record["custom_id"], record.get("error")))

        print(f"Chunk {i} ({bid}): parsed {len(labels)} labels so far")

    print(f"\nTotal labeled: {len(labels)}")
    print(f"Malformed/unrecognized responses: {len(malformed)}")
    print(f"Per-request errors: {len(request_errors)}")

    if malformed:
        print("\nSample malformed responses:")
        for cid, raw in malformed[:10]:
            print(f"  {cid}: {raw!r}")

    if request_errors:
        print("\nSample request errors:")
        for cid, err in request_errors[:10]:
            print(f"  {cid}: {err}")

    # --- merge onto source CSV ---
    df = pd.read_csv(SOURCE_CSV)
    df["source_id"] = df["source_id"].astype(str)
    df["emotion_label"] = df["source_id"].map(labels)

    unmatched = df["emotion_label"].isna().sum()
    print(f"\nRows in CSV with no label after merge: {unmatched} / {len(df)}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    print("\nLabel distribution:")
    print(df["emotion_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()