"""
HinEmo — Download Output Files for All Completed Batches
================================================================
Standalone — checks every batch ID in _batch_job_ids.txt, and for any
that are 'completed' but don't yet have a local output file downloaded,
downloads it. Safe to run anytime, including after 6d finishes overnight.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID_FILE = "data/interim/_batch_job_ids.txt"
OUTPUT_TEMPLATE = "data/interim/sentimix_batch_output_part{n}.jsonl"


def main():
    with open(BATCH_ID_FILE) as f:
        batch_ids = [line.strip() for line in f if line.strip()]

    print(f"Checking {len(batch_ids)} batch job(s)...\n")

    for i, batch_id in enumerate(batch_ids, start=1):
        batch = client.batches.retrieve(batch_id)
        print(f"[{i}] {batch_id}: {batch.status}")

        if batch.status == "completed" and batch.output_file_id:
            out_path = OUTPUT_TEMPLATE.format(n=batch_id[-6:])  # unique suffix per batch
            if os.path.exists(out_path):
                print(f"  already downloaded -> {out_path}")
                continue
            content = client.files.content(batch.output_file_id)
            with open(out_path, "wb") as f:
                f.write(content.content)
            print(f"  downloaded -> {out_path}")
        elif batch.status in ("failed", "expired", "cancelled"):
            print(f"  [!] not usable (status: {batch.status})")
        else:
            print(f"  still in progress")


if __name__ == "__main__":
    main()