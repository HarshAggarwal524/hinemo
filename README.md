# HinEmo Data Collection Pipeline (Ekman version)

## Why this differs from the original plan

Your original Segment 1 targeted fresh Twitter/X scraping via `snscrape`. That's
no longer viable: `snscrape`, `Twint`, and public Nitter instances all broke
between 2023-2024 when X locked down guest-token access, and haven't come back.
Free, unauthenticated Twitter scraping is effectively dead as of 2026.

The existing large emotion-labeled Hinglish dataset (Wadhawan & Aggarwal 2021,
151k tweets) is also not directly usable — it only distributes `tweet_id` +
one-hot label, not text, per Twitter's redistribution terms. Hydrating a decade
of tweet IDs through a paid API would be expensive and lossy (many are deleted).

## What we're using instead

1. **SentiMix (Patwa et al. 2020)** — the same corpus Paper 1 used. Real text,
   real gold word-level Hindi/English tags, ~20k tweets. Free, no scraping
   needed. Only missing piece: emotion labels (it only has sentiment).

2. **Reddit via PRAW** — Reddit's official API is still free for
   non-commercial/research use in 2026 (100 req/min). Used to top up whichever
   emotion classes are thin after the SentiMix base, using the same
   topic-targeting logic your plan used for Twitter hashtags.

3. **GPT-4o-mini labeling** — unchanged from your Step 1.4, just retargeted to
   Ekman's six (anger, joy, sadness, fear, surprise, disgust) instead of
   Plutchik's six, since Ekman is what both source corpora were built around.
   This also removes the "trust" problem — no existing Hinglish corpus has a
   trust label, and Reddit-sourcing 5,000 clean trust-emotion examples from
   scratch would have been the single hardest part of the whole collection.

## Run order

```
1_reddit_collector.py       # needs Reddit API creds (free, reddit.com/prefs/apps)
                             # targets 11k raw items/emotion (~66k total raw)
                             # check the per-emotion counts it prints at the end
1b_twitter_topup.py          # OPTIONAL — only for classes that fell short above
                             # (~$0.00015/tweet via a paid gateway; expect this
                             # to mainly be needed for disgust and fear)
2_load_sentimix_base.py     # needs your Paper 1 SentiMix files, or re-download
3_label_emotions_gpt.py     # needs OPENAI_API_KEY — run on Reddit + Twitter
                             # top-up + SentiMix outputs
4_clean_and_balance.py      # needs IndicLID wired in for the Reddit/Twitter
                             # portion (SentiMix rows already have gold tags)
```

## Raw volume target

Final balanced target: 5,000/class × 6 Ekman classes = 30,000.
Raw collection target: ~60-70k total (≈11k/emotion), following the same
~50% survival-through-cleaning assumption your original Step 1.1 used for
Twitter (dedup, min-length, 20-80% code-mix ratio, spam removal all cost you
volume). Don't expect Reddit alone to hit 11k/class reliably — check the
per-emotion printout from `1_reddit_collector.py` and use `1b_twitter_topup.py`
to fill whatever's short, most likely `disgust` and `fear`.

## Known gaps you'll need to fill in

- `4_clean_and_balance.py` has a stub `tag_language_indiclid()` — plug in
  your actual IndicLID call per Step 1.3.
- Reddit topic queries in `1_reddit_collector.py` are a reasonable starting
  set but you should run them, check per-class yield, and add more
  subreddits/queries for whichever classes come up short before moving to
  GPT labeling — cheaper to fix at collection time than after labeling.
- Expect `disgust` and `fear` to be the hardest classes to hit 5,000 clean
  examples for. Budget extra Reddit query rounds for those two.
- Manual validation (Step 1.5, Cohen's Kappa) still applies exactly as
  written in your original plan — nothing about the source change affects
  the validation methodology.
