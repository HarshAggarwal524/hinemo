"""
HinEmo — Batch API Labeling, Submit Job (targeted round 2: sadness/disgust)
==================================================================================
Same prompt/model as every other batch labeling job. Chunks automatically
if the file exceeds the 2M enqueued-token ceiling.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_PATH = "data/interim/round2_mixed_stage3.csv"
BATCH_INPUT_TEMPLATE = "data/interim/round2_batch_input_part{n}.jsonl"
BATCH_ID_FILE = "data/interim/_batch_job_ids_round2.txt"

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
    prompt_chars = len(PROMPT_TEMPLATE) + len(text)
    return prompt_chars / 4 + 10


def chunk_dataframe(df):
    chunks = []
    current_chunk = []
    current_tokens = 0
    for _, row in df.iterrows():
        text = str(row["text_clean"])[:1000]
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
            text = str(row["text_clean"])[:1000]
            request = {
                "custom_id": str(row["source_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5.4-nano",
                    "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
                    "temperature": 0,
                    "max_completion_tokens": 10,
                },
            }
            f.write(json.dumps(request) + "\n")


import time  # add this import at the top of the file

def wait_for_capacity(poll_interval=30):
    while True:
        active = [b for b in client.batches.list(limit=100).data
                  if b.status in ("validating", "in_progress", "finalizing")]
        if not active:
            return
        print(f"  Waiting for capacity — {len(active)} batch(es) still active, rechecking in {poll_interval}s")
        time.sleep(poll_interval)


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    if df["source_id"].duplicated().any():
        df = df.drop_duplicates(subset=["source_id"])

    chunks = chunk_dataframe(df)
    print(f"Split into {len(chunks)} chunk(s)")

    # clear/start the ID file fresh for this run
    open(BATCH_ID_FILE, "w").close()

    for i, chunk_rows in enumerate(chunks, start=1):
        wait_for_capacity()  # don't fire the next chunk until budget is free

        chunk_path = BATCH_INPUT_TEMPLATE.format(n=i)
        build_batch_file(chunk_rows, chunk_path)
        print(f"\nChunk {i}/{len(chunks)}: {len(chunk_rows)} requests -> {chunk_path}")

        batch_input_file = client.files.create(file=open(chunk_path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Submitted: {batch.id} ({batch.status})")

        with open(BATCH_ID_FILE, "a") as f:
            f.write(batch.id + "\n")

    print(f"\nDone — batch IDs saved to {BATCH_ID_FILE}")


if __name__ == "__main__":
    main()