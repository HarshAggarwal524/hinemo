"""
HinEmo — Resubmit teaser5k Chunk 2 (the one that hit the token ceiling)
==============================================================================
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

CHUNK_PATH = "data/interim/teaser5k_batch_input_part2.jsonl"
BATCH_ID_FILE = "data/interim/_batch_job_ids_teaser5k.txt"


def main():
    batch_input_file = client.files.create(file=open(CHUNK_PATH, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Submitted: {batch.id} ({batch.status})")

    with open(BATCH_ID_FILE, "a") as f:
        f.write(batch.id + "\n")
    print(f"Appended to {BATCH_ID_FILE}")


if __name__ == "__main__":
    main()