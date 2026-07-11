"""
HinEmo Data Collection — Reddit Collector (Ekman version)
===========================================================
Collects Hinglish-candidate posts/comments from Indian subreddits, topic-targeted
per Ekman emotion (anger, joy, sadness, fear, surprise, disgust).

WHY REDDIT: Free-tier Twitter scraping (snscrape/Twint/Nitter) is dead as of
2023-2024. Reddit's official API via PRAW is still free for non-commercial /
research use (100 req/min, no dollar cost) as of 2026. This is the most
realistic free source of fresh, topically-targeted code-mixed text.

SETUP:
  pip install praw
  1. Go to https://www.reddit.com/prefs/apps
  2. Create a "script" type app -> get client_id + client_secret
  3. Fill in the credentials below (or set as env vars)

NOTE: Reddit's free tier caps most listing endpoints around ~1000 items per
query. Run this across many query/subreddit combinations (already set up below)
rather than expecting one giant pull.
"""

import praw
import pandas as pd
import time
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# CREDENTIALS — fill these in, or export as environment variables
# ---------------------------------------------------------------------------
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT = "HinEmoResearchCollector/1.0 (by u/YOUR_USERNAME)"

# ---------------------------------------------------------------------------
# TOPIC MAP — mirrors the original Twitter hashtag targeting strategy,
# adapted to subreddits + search queries likely to surface Hinglish text.
# Ekman's 6: anger, joy(happiness), sadness, fear, surprise, disgust
# ---------------------------------------------------------------------------
EMOTION_TARGETS = {
    "anger": {
        "subreddits": ["india", "IndiaSpeaks", "unitedstatesofindia", "bakchodi",
                       "IndianDankMemes", "developersIndia", "delhi", "mumbai"],
        "queries": ["farmer protest", "traffic rant", "corrupt", "cricket loss reaction",
                    "politician response", "railway complaint", "auto driver fight",
                    "gully rage", "petrol price bakwaas", "landlord problem",
                    "boss toxic office", "reservation debate angry"],
    },
    "happiness": {
        "subreddits": ["india", "bollywood", "IndianDankMemes", "CricketShitpost",
                       "developersIndia", "delhi", "mumbai", "IndianTeenagers"],
        "queries": ["diwali celebration", "cricket win reaction", "wedding season",
                    "promotion happy", "festival vibes", "family reunion",
                    "college fest masti", "shaadi mubarak", "ipl jeet gaye",
                    "job offer khushi", "new phone happy", "reunion dosti"],
    },
    "sadness": {
        "subreddits": ["india", "IndianTeenagers", "developersIndia", "mumbai", "delhi"],
        "queries": ["RIP tribute", "missing home", "job loss", "breakup Hindi",
                    "loneliness hostel", "farmer suicide news", "pet death dukh",
                    "parents fight ghar", "exam fail udaas", "friend dur chala gaya",
                    "pollution health sad", "flood disaster dukhad"],
    },
    "fear": {
        "subreddits": ["india", "IndiaSpeaks", "developersIndia", "IndianTeenagers",
                       "TwoXIndia", "delhi", "mumbai"],
        "queries": ["layoff fear", "exam anxiety", "health scare", "crime news area",
                    "job interview nervous", "safety concern women", "dar lag raha",
                    "chinta ho rahi hai", "night akela dar", "hospital report tension",
                    "police case daraya", "earthquake bhukamp dar", "loan emi tension"],
    },
    "surprise": {
        "subreddits": ["india", "IndianDankMemes", "bollywood", "CricketShitpost",
                       "developersIndia"],
        "queries": ["shocking news", "unexpected result", "plot twist", "cant believe",
                    "viral video reaction", "exam result shock", "achanak hua",
                    "kaise ho sakta hai", "last ball six shocking", "surprise party pataya",
                    "election result chauk gaya"],
    },
    "disgust": {
        "subreddits": ["india", "IndiaSpeaks", "bakchodi", "delhi", "mumbai",
                       "developersIndia"],
        "queries": ["disgusting behavior", "spit paan", "corruption disgust",
                    "gross food hygiene", "harassment reaction", "cringe ad",
                    "ghinauna kaam", "gandagi sadak", "bribe maang raha ghin",
                    "molestation reaction disgust", "adulterated food gross",
                    "public toilet gross india"],
    },
}

POSTS_PER_QUERY = 100          # Reddit search soft-caps around ~1000/query; keep modest per call
COMMENTS_PER_POST = 20
RAW_TARGET_PER_EMOTION = 11000  # -> ~66k raw total -> nets ~30k after cleaning at ~50% survival

# Looping over sort + time_filter combos returns genuinely different result
# sets from Reddit's search (not just re-fetching the same cap), which is
# how we multiply real yield instead of hitting the same ~1000-item ceiling
# over and over.
SORT_MODES = ["relevance", "top", "new"]
TIME_FILTERS = ["all", "year", "month"]


def collect_for_emotion(reddit, emotion, targets):
    rows = []
    seen_ids = set()

    for sort_mode in SORT_MODES:
        for time_filter in TIME_FILTERS:
            if len(rows) >= RAW_TARGET_PER_EMOTION:
                break
            for subreddit_name in targets["subreddits"]:
                if len(rows) >= RAW_TARGET_PER_EMOTION:
                    break
                subreddit = reddit.subreddit(subreddit_name)
                for query in targets["queries"]:
                    if len(rows) >= RAW_TARGET_PER_EMOTION:
                        break
                    try:
                        results = subreddit.search(
                            query, limit=POSTS_PER_QUERY,
                            sort=sort_mode, time_filter=time_filter,
                        )
                        for submission in results:
                            if submission.id not in seen_ids:
                                seen_ids.add(submission.id)
                                text = (submission.title or "") + " " + (submission.selftext or "")
                                if text.strip():
                                    rows.append({
                                        "source_id": submission.id,
                                        "source": "reddit_post",
                                        "subreddit": subreddit_name,
                                        "query": query,
                                        "sort_mode": sort_mode,
                                        "time_filter": time_filter,
                                        "target_emotion": emotion,
                                        "text": text.strip(),
                                        "created_utc": submission.created_utc,
                                    })

                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments[:COMMENTS_PER_POST]:
                                if comment.id in seen_ids:
                                    continue
                                seen_ids.add(comment.id)
                                if comment.body and comment.body not in ("[deleted]", "[removed]"):
                                    rows.append({
                                        "source_id": comment.id,
                                        "source": "reddit_comment",
                                        "subreddit": subreddit_name,
                                        "query": query,
                                        "sort_mode": sort_mode,
                                        "time_filter": time_filter,
                                        "target_emotion": emotion,
                                        "text": comment.body.strip(),
                                        "created_utc": comment.created_utc,
                                    })
                        time.sleep(0.5)  # be polite, stay well under rate limit
                    except Exception as e:
                        print(f"  [!] {subreddit_name} / '{query}' / {sort_mode}/{time_filter}: {e}")
                        time.sleep(2)

        print(f"  ... {emotion}: {len(rows)} raw items so far (target {RAW_TARGET_PER_EMOTION})")

    if len(rows) < RAW_TARGET_PER_EMOTION:
        print(f"  [!] {emotion} fell short: {len(rows)}/{RAW_TARGET_PER_EMOTION}. "
              f"Add more subreddits/queries to EMOTION_TARGETS, or plan to top "
              f"this class up via the paid Twitter gateway fallback.")

    return rows


def main():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    all_rows = []
    for emotion, targets in EMOTION_TARGETS.items():
        print(f"Collecting for target emotion: {emotion}")
        rows = collect_for_emotion(reddit, emotion, targets)
        print(f"  -> {len(rows)} raw items")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"])
    out_path = f"reddit_raw_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} unique items to {out_path}")
    print("NOTE: 'target_emotion' is a collection HINT from topic-targeting, not a")
    print("gold label. Every item still goes through GPT-4o-mini labeling in step 2.")


if __name__ == "__main__":
    main()
