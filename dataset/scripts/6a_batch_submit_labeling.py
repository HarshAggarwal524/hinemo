"""
HinEmo — Batch API Labeling, Submit Job (SentiMix data) — chunked version
=============================================================
Same prompt/model/output quality as before — this version just splits the
requests across multiple smaller batch jobs so no single job trips the
per-org "enqueued token limit" for gpt-5.4-nano (2,000,000 tokens).

Splitting does NOT increase cost. OpenAI's Batch API bills per token
processed with the same 50% batch discount no matter how many separate
batch jobs you submit — this just avoids the token_limit_exceeded error.

Input:  data/raw/twitter_takenfromsemevalpaper.csv (read-only, unchanged)
Output: data/interim/sentimix_batch_input_part{N}.jsonl (one file per chunk)
        data/interim/_batch_job_ids.txt (one batch ID per line, this run)
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_PATH = "data/raw/twitter_takenfromsemevalpaper.csv"
BATCH_INPUT_JSONL_TEMPLATE = "data/interim/sentimix_batch_input_part{n}.jsonl"
BATCH_ID_FILE = "data/interim/_batch_job_ids.txt"  # now one ID per line

# Stay comfortably under the 2,000,000 enqueued-token org limit per batch,
# leaving headroom in case other batch jobs are already in flight.
TOKEN_BUDGET_PER_CHUNK = 1_500_000

PROMPT_TEMPLATE = """You are an emotion classifier for code-mixed Hinglish social
media comments. Classify the emotion expressed in the comment below.

Categories and how to tell them apart:
- anger: direct outrage or hostility at a specific action/person — "I'm furious
  this happened." Energetic, confrontational.
- disgust: moral condemnation, contempt, or revulsion — "this is shameful/beneath
  contempt," reactions to corruption, hypocrisy, or gross behavior. More about
  judgment than heat.
- fear: worry about something that HASN'T happened yet or an ongoing threat —
  anxiety about the future, safety concerns, dread.
- sadness: grief or despair about something that HAS happened or already IS —
  loss, disappointment, mourning, sympathy for suffering already occurring.
- joy: happiness, celebration, excitement, affection, pride.
- surprise: an unexpected reaction — ONLY use this if shock/disbelief is the
  MAIN feeling with no strong secondary emotion attached. If the comment is
  shocked AND clearly angry/happy/etc., classify by that other emotion instead.
- neutral: no clear emotion — purely factual, a question, or a mild/generic
  reaction ("nice video", "good point") without strong feeling behind it.

Handling sarcasm and mocking laughter: If a comment uses laughing emoji (😂🤣)
or "lol" while mocking, insulting, or accusing someone, the underlying emotion
is usually disgust or anger, NOT joy — the laughter is contempt, not happiness.
Only classify as joy if the positive feeling is genuine, not mocking. Do not
assume political content is automatically sarcastic — many political comments
are sincere.

Handling mixed emotions: if two emotions both seem present, pick whichever is
the PRIMARY driver of the comment's tone, not a secondary undertone.

Examples:
Comment: "Bhai ye toh bahot hi shameful hai, sharam karo"
Emotion: disgust

Comment: "Kal exam hai, kuch samajh nahi aa raha, dar lag raha hai"
Emotion: fear

Comment: "Uski death ki news sunke bahut dukh hua"
Emotion: sadness

Comment: "chor ho 😂😂😂" (mocking, accusing someone of being a thief)
Emotion: disgust

Comment: "Itna acha result aayega socha nahi tha!"
Emotion: joy

Comment: "Kaunse video mein ye scene tha?"
Emotion: neutral

Comment: "Wait what, ye kaise ho gaya"
Emotion: surprise

Now classify this comment. Respond with only one word from the category list
above, nothing else.

Comment: {text}
Emotion:"""


def estimate_tokens(text):
    """Rough estimate (~4 chars/token) of the enqueued tokens for one request:
    the fixed prompt template + this row's text + the max completion tokens
    we allow. Doesn't need to be exact — just good enough to pack chunks
    safely under the limit."""
    prompt_chars = len(PROMPT_TEMPLATE) + len(text)
    input_tokens = prompt_chars / 4
    return input_tokens + 10  # + max_completion_tokens


def chunk_dataframe(df):
    """Greedily pack rows into chunks so each chunk's estimated enqueued
    tokens stays under TOKEN_BUDGET_PER_CHUNK."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for _, row in df.iterrows():
        text = str(row["cleaned_text"])[:1000]
        row_tokens = estimate_tokens(text)

        if current_chunk and current_tokens + row_tokens > TOKEN_BUDGET_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(row)
        current_tokens += row_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_batch_file(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            text = str(row["cleaned_text"])[:1000]
            request = {
                "custom_id": str(row["source_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4-nano",
                    "messages": [
                        {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                    ],
                    "temperature": 0,
                    "max_completion_tokens": 10,
                },
            }
            f.write(json.dumps(request) + "\n")


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    if df["source_id"].duplicated().any():
        print("[!] Warning: duplicate source_id values found — custom_id must be unique.")
        print("    Dropping duplicates before building batch files.")
        df = df.drop_duplicates(subset=["source_id"])

    chunks = chunk_dataframe(df)
    print(f"Split {len(df)} rows into {len(chunks)} chunk(s) to stay under the token limit")

    os.makedirs(os.path.dirname(BATCH_ID_FILE), exist_ok=True)
    batch_ids = []

    for i, chunk_rows in enumerate(chunks, start=1):
        chunk_path = BATCH_INPUT_JSONL_TEMPLATE.format(n=i)
        build_batch_file(chunk_rows, chunk_path)
        print(f"\nChunk {i}/{len(chunks)}: wrote {len(chunk_rows)} requests to {chunk_path}")

        batch_input_file = client.files.create(
            file=open(chunk_path, "rb"),
            purpose="batch",
        )
        print(f"  Uploaded file: {batch_input_file.id}")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Batch job submitted: {batch.id}")
        print(f"  Status: {batch.status}")

        batch_ids.append(batch.id)

    with open(BATCH_ID_FILE, "w") as f:
        f.write("\n".join(batch_ids) + "\n")

    print(f"\nSaved {len(batch_ids)} batch ID(s) to {BATCH_ID_FILE}")
    print("Run scripts/6b_batch_check_and_retrieve.py to check status and get results.")


if __name__ == "__main__":
    main()