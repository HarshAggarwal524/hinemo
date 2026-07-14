"""
HinEmo — Batched vs Single-Item Labeling Comparison
========================================================
Takes a random sample from the already-batch-labeled output and re-labels
the SAME comments one-at-a-time (no batching), then compares the two sets
of labels to check whether batching introduced cross-comment contamination
or other quality loss.

Run this AFTER 3_label_emotions_gpt.py has finished producing
data/interim/youtube_labeled_stage5.csv.
"""

import os
import time
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BATCH_LABELED_PATH = "data/interim/youtube_labeled_stage5.csv"
OUTPUT_PATH = "data/interim/labeling_comparison_sample.csv"

SAMPLE_SIZE = 1000
VALID_EMOTIONS = {"anger", "joy", "sadness", "fear", "surprise", "disgust", "neutral"}

# Same prompt logic as the batched version, just single-item — this is the
# fair comparison: identical categories/instructions, only difference is
# whether the comment was judged alone or alongside 24 others.
SINGLE_PROMPT_TEMPLATE = """You are an emotion classifier for code-mixed Hinglish social
media comments. Classify the emotion expressed in the comment below.

Categories and how to tell them apart:
- anger: direct outrage or hostility at a specific action/person.
- disgust: moral condemnation, contempt, or revulsion — reactions to corruption,
  hypocrisy, shameful behavior.
- fear: worry about something that HASN'T happened yet — anxiety, dread, threat.
- sadness: grief or despair about something that HAS happened — loss, disappointment.
- joy: happiness, celebration, excitement, affection, pride.
- surprise: shock/disbelief as the MAIN feeling, with no strong secondary emotion.
- neutral: no clear emotion — factual, a question, or mild/generic reaction.

If sarcastic/mocking, classify by the real underlying feeling. Do not assume
political content is automatically sarcastic. If two emotions seem present,
pick the primary driver.

Respond with only one word from the category list above, nothing else.

Comment: {text}
Emotion:"""


def label_single(text):
    text = str(text)[:1000]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": SINGLE_PROMPT_TEMPLATE.format(text=text)}],
            temperature=0,
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        return label if label in VALID_EMOTIONS else "unknown"
    except Exception as e:
        print(f"  [!] error: {e}")
        time.sleep(2)
        return "error"


def main():
    df = pd.read_csv(BATCH_LABELED_PATH)
    print(f"Loaded {len(df)} batch-labeled rows")

    random.seed(42)
    sample_idx = random.sample(range(len(df)), min(SAMPLE_SIZE, len(df)))
    sample = df.iloc[sample_idx].copy()
    print(f"Sampled {len(sample)} rows for comparison")

    single_labels = []
    for i, text in enumerate(sample["text_clean"]):
        single_labels.append(label_single(text))
        if (i + 1) % 100 == 0:
            print(f"  labeled {i + 1}/{len(sample)}")

    sample["emotion_single"] = single_labels
    sample = sample.rename(columns={"emotion": "emotion_batch"})

    sample["agree"] = sample["emotion_batch"] == sample["emotion_single"]
    agreement_rate = sample["agree"].mean()

    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved comparison to {OUTPUT_PATH}")
    print(f"\nAgreement rate: {agreement_rate:.1%}")
    print(f"\nDisagreements by batch-label -> single-label:")
    disagreements = sample[~sample["agree"]]
    print(disagreements.groupby(["emotion_batch", "emotion_single"]).size().sort_values(ascending=False))


if __name__ == "__main__":
    main()