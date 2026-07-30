"""
HinEmo — Resubmit any round-2 chunks that failed with token_limit_exceeded.
Uses the local jsonl files that were already built, no need to rebuild from CSV.
Waits for zero active batches before each resubmission to avoid re-triggering
the same enqueued-token limit.
"""

import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID_FILE = "data/interim/_batch_job_ids_round2.txt"
BATCH_INPUT_TEMPLATE = "data/interim/round2_batch_input_part{n}.jsonl"


def wait_for_capacity(poll_interval=30):
    while True:
        active = [b for b in client.batches.list(limit=100).data
                  if b.status in ("validating", "in_progress", "finalizing")]
        if not active:
            return
        print(f"  Waiting for capacity — {len(active)} active, retrying in {poll_interval}s")
        time.sleep(poll_interval)


def main():
    with open(BATCH_ID_FILE) as f:
        old_ids = [line.strip() for line in f if line.strip()]

    new_ids = []
    for i, bid in enumerate(old_ids, start=1):
        b = client.batches.retrieve(bid)
        if b.status != "failed":
            print(f"Chunk {i} ({bid}): {b.status} — leaving as is")
            new_ids.append(bid)
            continue

        print(f"Chunk {i} ({bid}): failed — resubmitting part{i}")
        wait_for_capacity()

        f_in = client.files.create(
            file=open(BATCH_INPUT_TEMPLATE.format(n=i), "rb"), purpose="batch"
        )
        batch = client.batches.create(
            input_file_id=f_in.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Resubmitted as: {batch.id} ({batch.status})")
        new_ids.append(batch.id)

    with open(BATCH_ID_FILE, "w") as f:
        f.write("\n".join(new_ids) + "\n")
    print(f"\nUpdated {BATCH_ID_FILE}")


if __name__ == "__main__":
    main()