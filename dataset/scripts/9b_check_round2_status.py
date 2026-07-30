"""
HinEmo — Full Status Check, Round 2 Batch Jobs
====================================================
Checks EVERY batch ID submitted for round 2 (not just currently-active
ones) and reports its final outcome — completed, failed (with error
detail), or still in progress.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID_FILE = "data/interim/_batch_job_ids_round2.txt"


def main():
    with open(BATCH_ID_FILE) as f:
        batch_ids = [line.strip() for line in f if line.strip()]

    print(f"Checking {len(batch_ids)} batch job(s) submitted for round 2:\n")

    for i, batch_id in enumerate(batch_ids, start=1):
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        progress = f"{counts.completed}/{counts.total} completed, {counts.failed} failed" if counts else "?"

        print(f"[{i}] {batch_id}")
        print(f"    status: {batch.status}")
        print(f"    progress: {progress}")

        if batch.status == "failed" and batch.errors:
            for err in batch.errors.data:
                print(f"    error: {err.code} — {err.message}")

        print()


if __name__ == "__main__":
    main()