"""
HinEmo Data Collection — Twitter Fallback (paid gateway, for topping up thin classes)
=========================================================================================
Use this ONLY for classes that fell short after running 1_reddit_collector.py
(check the raw counts it prints per emotion). Don't run this for all 6 classes
by default — it costs money per call, Reddit already covers most classes free.

Uses a third-party gateway (e.g. TwitterAPI.io) rather than the official X API,
since the official API starts at $100/mo minimum and this is far cheaper for
research-scale volume (~$0.00015/tweet as of mid-2026 — verify current pricing
before running, rates change).

At $0.00015/tweet, topping up e.g. disgust + fear by 8,000 tweets each
(16,000 total) costs roughly $2.40. Topping up all 6 classes fully from
scratch (66k tweets) would cost roughly $10.

SETUP:
  pip install requests
  Sign up for a gateway (TwitterAPI.io or similar), get an API key.
  export TWITTER_GATEWAY_API_KEY=your_key

NOTE: this script is a template — the exact endpoint/params depend on which
gateway you pick. Check their current docs before running; gateway APIs in
this space change fast. The structure below (search by query, paginate,
collect text) is standard across most of them.
"""

import os
import time
import requests
import pandas as pd

API_KEY = os.environ.get("TWITTER_GATEWAY_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://api.example-gateway.com/v1/search"  # <-- replace with your chosen gateway's real endpoint

# Only fill in the classes that actually came up short from Reddit.
# Check the raw counts 1_reddit_collector.py printed before deciding what
# goes here — don't just copy this wholesale.
TOPUP_TARGETS = {
    "disgust": {
        "target_count": 8000,
        "queries": ["ghinauna India", "gross food india twitter", "corruption disgust india",
                    "spit paan disgusting", "gandagi india complaint"],
    },
    "fear": {
        "target_count": 8000,
        "queries": ["dar lag raha hai", "safety scare india twitter", "layoff fear india",
                    "exam anxiety hindi", "crime news area scare"],
    },
}

TWEETS_PER_REQUEST = 100  # gateway-dependent, check your provider's max page size


def fetch_tweets(query, target_count):
    collected = []
    cursor = None
    while len(collected) < target_count:
        params = {
            "query": query,
            "lang": "hi",  # bias toward Hindi/mixed, most gateways support this
            "count": TWEETS_PER_REQUEST,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(
                BASE_URL,
                headers={"X-API-Key": API_KEY},
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [!] request failed for '{query}': {e}")
            break

        tweets = data.get("tweets", [])
        if not tweets:
            break

        for t in tweets:
            collected.append({
                "source_id": t.get("id"),
                "source": "twitter_gateway",
                "query": query,
                "text": t.get("text", ""),
                "created_at": t.get("created_at"),
            })

        cursor = data.get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)

    return collected[:target_count]


def main():
    all_rows = []
    for emotion, cfg in TOPUP_TARGETS.items():
        print(f"Topping up: {emotion} (target {cfg['target_count']})")
        per_query_target = cfg["target_count"] // len(cfg["queries"]) + 1
        emotion_rows = []
        for query in cfg["queries"]:
            rows = fetch_tweets(query, per_query_target)
            print(f"  '{query}' -> {len(rows)} tweets")
            for r in rows:
                r["target_emotion"] = emotion
            emotion_rows.extend(rows)
        print(f"  {emotion} total: {len(emotion_rows)}")
        all_rows.extend(emotion_rows)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"])
    df.to_csv("twitter_topup_raw.csv", index=False)
    print(f"\nSaved {len(df)} unique tweets to twitter_topup_raw.csv")
    print("Feed this through 3_label_emotions_gpt.py same as the Reddit output —")
    print("'target_emotion' here is still just a collection hint, not gold label.")


if __name__ == "__main__":
    main()
