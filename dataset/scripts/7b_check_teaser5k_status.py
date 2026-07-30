"""
HinEmo — Check Status of the teaser5k Batch Job
======================================================
Quick status check — safe to rerun anytime while waiting.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID_FILE = "data/interim/_batch_job_ids_teaser5k.txt"

def main():
    with open(BATCH_ID_FILE) as f:
        batch_id = f.read().strip()

    batch = client.batches.retrieve(batch_id)
    print(f"Batch ID: {batch_id}")
    print(f"Status: {batch.status}")
    if batch.request_counts:
        counts = batch.request_counts
        print(f"Progress: {counts.completed}/{counts.total} completed, {counts.failed} failed")

    if batch.status == "completed":
        print(f"\nOutput file ready: {batch.output_file_id}")
        print("Run scripts/7c_parse_teaser5k_results.py to download and parse results.")
    
    if batch.status == "failed":
        print(f"\nErrors: {batch.errors}")
        if batch.error_file_id:
            content = client.files.content(batch.error_file_id)
            print(content.content.decode())


if __name__ == "__main__":
    main()