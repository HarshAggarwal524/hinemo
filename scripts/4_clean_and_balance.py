"""
HinEmo Data Collection — Clean, Tag, and Balance to 30k (5k x 6 Ekman classes)
=================================================================================
1. Merges the labeled SentiMix + Reddit data
2. Applies your original cleaning rules (Step 1.2): dedup, min length,
   code-mixing ratio 20-80%, spam/URL removal
3. Runs IndicLID on the Reddit portion only (SentiMix already has gold tags)
4. Balances each Ekman class down to a target count (default 5000)
5. Splits 70/15/15 train/val/test with stratification
6. Saves hinemo_train.csv / hinemo_val.csv / hinemo_test.csv

SETUP:
  pip install ai4bharat-transliteration indic-nlp-library scikit-learn pandas
  (IndicLID: https://github.com/AI4Bharat/IndicLID — follow their repo setup;
   swap in whatever language-ID tool you actually installed if different)
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_PER_CLASS = 5000
MIN_TOKENS = 5
HINDI_RATIO_MIN, HINDI_RATIO_MAX = 0.20, 0.80

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")


# ---------------------------------------------------------------------------
# Step 1.2 — Cleaning
# ---------------------------------------------------------------------------
def basic_clean(text):
    text = URL_RE.sub("", str(text))
    text = MENTION_RE.sub("", text)
    return text.strip()


def token_count(text):
    return len(text.split())


def hindi_ratio_from_tags(lang_tags):
    """For rows that already have gold/predicted per-token lang tags."""
    if not lang_tags:
        return None
    tags = lang_tags if isinstance(lang_tags, list) else eval(lang_tags)
    if not tags:
        return None
    hindi = sum(1 for t in tags if str(t).lower() in ("hin", "hi"))
    return hindi / len(tags)


def tag_language_indiclid(texts):
    """
    Placeholder wrapper — plug in your actual IndicLID call here.
    Should return a list of per-token tag lists, one per input text,
    using labels compatible with 'Hin' / 'Eng' / 'O' as in Step 1.3.

    Example integration point:
        from indiclid import IndicLID
        model = IndicLID()
        return [model.predict_tokens(t) for t in texts]
    """
    raise NotImplementedError(
        "Wire up IndicLID here per Step 1.3 of your plan before running "
        "on the Reddit-sourced rows."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    frames = []
    for path in ["sentimix_labeled.csv", "reddit_labeled.csv"]:
        try:
            frames.append(pd.read_csv(path))
        except FileNotFoundError:
            print(f"[!] {path} not found, skipping")

    if not frames:
        print("No labeled files found — run steps 1-3 first.")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} total rows")

    # --- Clean ---
    df["text"] = df["text"].apply(basic_clean)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].apply(token_count) >= MIN_TOKENS]
    print(f"After dedup + min-length: {len(df)}")

    # --- Language tag / code-mix ratio filter ---
    # Rows from SentiMix already have gold lang_tags -> use directly.
    if "lang_tags" in df.columns:
        df["hindi_ratio"] = df["lang_tags"].apply(
            lambda x: hindi_ratio_from_tags(x) if pd.notna(x) else None
        )
    else:
        df["hindi_ratio"] = None

    needs_tagging = df["hindi_ratio"].isna()
    print(f"{needs_tagging.sum()} rows need IndicLID tagging (Reddit-sourced)")
    if needs_tagging.sum() > 0:
        print("  -> wire up tag_language_indiclid() before running this on")
        print("     real data; for now these rows are excluded from the ratio filter.")
        # Once implemented:
        # tags = tag_language_indiclid(df.loc[needs_tagging, "text"].tolist())
        # df.loc[needs_tagging, "lang_tags"] = tags
        # df.loc[needs_tagging, "hindi_ratio"] = [hindi_ratio_from_tags(t) for t in tags]

    df = df[df["hindi_ratio"].between(HINDI_RATIO_MIN, HINDI_RATIO_MAX)]
    print(f"After code-mix ratio filter (20-80% Hindi): {len(df)}")

    # --- Emotion class check ---
    print("\nClass distribution before balancing:")
    print(df["emotion"].value_counts())

    # --- Balance ---
    balanced_frames = []
    for emotion, group in df.groupby("emotion"):
        if len(group) < TARGET_PER_CLASS:
            print(f"  [!] {emotion} has only {len(group)} rows (< {TARGET_PER_CLASS} target)")
            balanced_frames.append(group)
        else:
            balanced_frames.append(group.sample(n=TARGET_PER_CLASS, random_state=42))
    balanced = pd.concat(balanced_frames, ignore_index=True)
    print(f"\nBalanced dataset: {len(balanced)} rows")
    print(balanced["emotion"].value_counts())

    # --- Split 70/15/15, stratified ---
    train, temp = train_test_split(
        balanced, test_size=0.30, stratify=balanced["emotion"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["emotion"], random_state=42
    )

    train.to_csv("hinemo_train.csv", index=False)
    val.to_csv("hinemo_val.csv", index=False)
    test.to_csv("hinemo_test.csv", index=False)

    print(f"\nSaved: hinemo_train.csv ({len(train)}), "
          f"hinemo_val.csv ({len(val)}), hinemo_test.csv ({len(test)})")


if __name__ == "__main__":
    main()
