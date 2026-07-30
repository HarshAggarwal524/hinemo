"""
HinEmo Data Collection — Emotion Labeling (Ekman + neutral, single-item, GPT-5.4-nano)
============================================================================================
No batching — every comment gets its own isolated API call, avoiding any
cross-comment contamination risk. Concurrent via thread pool for speed,
checkpoints every 500 rows so a crash/interruption/rate-limit wall doesn't
lose progress — rerun the script and it resumes automatically.

Uses gpt-5.4-nano instead of gpt-4o-mini: no RPD (requests-per-day) cap on
this account tier, recommended by OpenAI specifically for classification
tasks, and benchmarks better than gpt-4o-mini at similar/lower cost.
"""

import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_PATH = "data/interim/youtube_codemixed_stage4.csv"
OUTPUT_PATH = "data/interim/youtube_labeled_stage5.csv"
CHECKPOINT_PATH = "data/interim/_checkpoint_labeling.csv"

CHUNK_SIZE = 500
MAX_WORKERS = 8          # reduced to stay under gpt-5.4-nano's 200k TPM given our long prompt

VALID_EMOTIONS = {"anger", "joy", "sadness", "fear", "surprise", "disgust", "neutral"}

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


def label_one(args):
    idx, text = args
    text = str(text)[:1000]
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
                temperature=0,
                max_completion_tokens=10,
            )
            label = resp.choices[0].message.content.strip().lower()
            return idx, (label if label in VALID_EMOTIONS else "unknown")
        except Exception as e:
            if "requests per day" in str(e).lower():
                # daily cap — no point retrying immediately, surface it clearly
                return idx, "RATE_LIMIT_DAY"
            if "tokens per day" in str(e).lower():
                return idx, "RATE_LIMIT_DAY"  # same stop-cleanly behavior as RPD
            if "tokens per min" in str(e).lower() and attempt < 4:
                time.sleep(3 * (attempt + 1))  # 3s, 6s, 9s, 12s — patient for short TPM windows
                continue
            if "rate" in str(e).lower() and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"  [!] API error on row {idx}: {e}")
            return idx, "error"
    return idx, "error"


def label_chunk_concurrent(chunk_items):
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(label_one, item) for item in chunk_items]
        for future in as_completed(futures):
            idx, label = future.result()
            results[idx] = label
    return results


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    already_labeled = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        already_labeled = len(checkpoint_df)
        print(f"Resuming from checkpoint: {already_labeled} already labeled")
        labels = list(checkpoint_df["emotion"])
    else:
        labels = []

    remaining = df.iloc[already_labeled:]
    remaining_items = list(zip(remaining.index, remaining["text_clean"]))

    start_time = time.time()
    hit_daily_limit = False

    for chunk_start in range(0, len(remaining_items), CHUNK_SIZE):
        chunk = remaining_items[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_results = label_chunk_concurrent(chunk)

        if any(v == "RATE_LIMIT_DAY" for v in chunk_results.values()):
            print("\n[!] Hit daily request-limit wall. Stopping here — your progress")
            print("    is saved in the checkpoint. Rerun this script tomorrow (or once")
            print("    your quota resets) to continue from exactly this point.")
            hit_daily_limit = True
            break

        for idx, _ in chunk:
            labels.append(chunk_results[idx])

        current_count = already_labeled + chunk_start + len(chunk)
        tmp = df.iloc[:current_count].copy()
        tmp["emotion"] = labels
        tmp.to_csv(CHECKPOINT_PATH, index=False)

        elapsed = time.time() - start_time
        rate = (chunk_start + len(chunk)) / elapsed if elapsed > 0 else 0
        remaining_count = len(remaining_items) - (chunk_start + len(chunk))
        eta_min = (remaining_count / rate / 60) if rate > 0 else 0
        error_count = labels.count("error") + labels.count("unknown")
        print(f"  checkpoint: {current_count}/{len(df)} labeled "
              f"({rate:.1f} rows/sec, ETA {eta_min:.0f} min, "
              f"{error_count} errors/unknowns so far)")

    if hit_daily_limit:
        print(f"\nStopped early — {len(labels)}/{len(df)} labeled so far this run.")
        return

    df["emotion"] = labels
    before = len(df)
    neutral_count = (df["emotion"] == "neutral").sum()
    unlabelable_count = df["emotion"].isin(["unknown", "error"]).sum()
    df = df[~df["emotion"].isin(["unknown", "error", "neutral"])]
    print(f"\nDropped {before - len(df)} rows total "
          f"({neutral_count} neutral, {unlabelable_count} unlabelable)")

    df.to_csv(OUTPUT_PATH, index=False)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"Saved {len(df)} labeled rows -> {OUTPUT_PATH}")
    print("\nLabel distribution:")
    print(df["emotion"].value_counts())


if __name__ == "__main__":
    main()