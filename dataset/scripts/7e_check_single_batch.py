"""
HinEmo — Check a Single Batch ID Directly
================================================
Quick one-off check for any specific batch ID — bypasses list() in case
a batch isn't showing up there for some reason.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BATCH_ID = "batch_6a5729aa383c8190b69d52b2f5e1c926"


def main():
    batch = client.batches.retrieve(BATCH_ID)
    print(f"Status: {batch.status}")
    print(f"Request counts: {batch.request_counts}")
    if batch.status == "failed":
        print(f"Errors: {batch.errors}")
    if batch.status == "completed":
        print(f"Output file: {batch.output_file_id}")


if __name__ == "__main__":
    main()