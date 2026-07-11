"""
HinEmo Data Collection — Emotion Labeling (Ekman, via GPT-4o-mini)
=====================================================================
Applies gold-standard-style silver labeling using GPT-4o-mini, exactly per
your plan's Step 1.4 (temperature=0 for reproducibility), but targeting
Ekman's six: anger, joy, sadness, fear, surprise, disgust.

Run this on BOTH the SentiMix base and the Reddit-collected data — every
row needs a real label; the "target_emotion" column from the Reddit
collector is only a topical hint, not ground truth.

SETUP:
  pip install openai
  export OPENAI_API_KEY=your_key
"""

import os
import time
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

VALID_EMOTIONS = {"anger", "joy", "sadness", "fear", "surprise", "disgust"}

PROMPT_TEMPLATE = """You are an emotion classifier for code-mixed Hinglish social
media text. Classify the following text into exactly one of
these emotions: anger, joy, sadness, fear, surprise, disgust.
Respond with only the emotion word, nothing else.

Text: {text}
Emotion:"""


def label_one(text):
    text = str(text)[:1000]  # guard against extreme-length outliers
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
            temperature=0,
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        return label if label in VALID_EMOTIONS else "unknown"
    except Exception as e:
        print(f"  [!] API error: {e}")
        time.sleep(2)
        return "error"


def label_dataframe(df, text_col="text", checkpoint_every=200, checkpoint_path=None):
    labels = []
    for i, text in enumerate(df[text_col]):
        labels.append(label_one(text))
        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            tmp = df.iloc[: i + 1].copy()
            tmp["emotion"] = labels
            tmp.to_csv(checkpoint_path, index=False)
            print(f"  checkpoint: {i + 1}/{len(df)} labeled")
    df = df.copy()
    df["emotion"] = labels
    return df


def main():
    inputs = [
        ("sentimix_base_with_lang_tags.csv", "sentimix_labeled.csv"),
        # add your reddit_raw_*.csv path here after running the collector
        # ("reddit_raw_YYYYMMDD.csv", "reddit_labeled.csv"),
    ]

    for in_path, out_path in inputs:
        if not os.path.exists(in_path):
            print(f"[!] skipping missing file: {in_path}")
            continue
        df = pd.read_csv(in_path)
        print(f"Labeling {len(df)} rows from {in_path} ...")
        labeled = label_dataframe(df, checkpoint_path=f"_checkpoint_{out_path}")

        before = len(labeled)
        labeled = labeled[~labeled["emotion"].isin(["unknown", "error"])]
        print(f"  dropped {before - len(labeled)} unlabelable rows")

        labeled.to_csv(out_path, index=False)
        print(f"  saved -> {out_path}")
        print(labeled["emotion"].value_counts())
        print()


if __name__ == "__main__":
    main()
