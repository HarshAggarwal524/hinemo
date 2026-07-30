"""
HinEmo — Resubmit Failed Batch Chunks, One at a Time
==========================================================
Submits chunks SEQUENTIALLY, waiting for each to finish before submitting
the next, so they can't collide with the org-wide in-flight token ceiling.
Reuses the existing input JSONL files already on disk — doesn't rebuild them.
"""

import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

FAILED_CHUNK_NUMBERS = [1, 3, 4, 5, 7]
INPUT_TEMPLATE = "data/interim/sentimix_batch_input_part{n}.jsonl"
BATCH_ID_FILE = "data/interim/_batch_job_ids.txt"
POLL_INTERVAL_SECONDS = 60


def submit_and_wait(chunk_path):
    batch_input_file = client.files.create(file=open(chunk_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Submitted: {batch.id}")

    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"    status: {batch.status}")
        if batch.status not in ("validating", "in_progress", "finalizing"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    return batch


def main():
    new_ids = []
    for n in FAILED_CHUNK_NUMBERS:
        chunk_path = INPUT_TEMPLATE.format(n=n)
        print(f"\nResubmitting chunk {n} ({chunk_path})...")
        batch = submit_and_wait(chunk_path)
        print(f"  Chunk {n} finished with status: {batch.status}")
        new_ids.append(batch.id)

    with open(BATCH_ID_FILE, "a") as f:
        for bid in new_ids:
            f.write(bid + "\n")

    print(f"\nAppended {len(new_ids)} new batch ID(s) to {BATCH_ID_FILE}")
    


if __name__ == "__main__":
    main()