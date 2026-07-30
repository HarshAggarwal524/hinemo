"""
HinEmo — List Currently In-Progress Batches
=================================================
Checks every batch job on the account and shows which ones are still
consuming the shared 2M enqueued-token budget right now.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def main():
    batches = client.batches.list(limit=100)

    active = [b for b in batches.data if b.status in ("validating", "in_progress", "finalizing")]

    if not active:
        print("No batches currently in progress — safe to submit new jobs.")
        return

    print(f"{len(active)} batch(es) still active:\n")
    for b in active:
        counts = b.request_counts
        progress = f"{counts.completed}/{counts.total}" if counts else "?"
        print(f"  {b.id}: {b.status} ({progress})")


if __name__ == "__main__":
    main()