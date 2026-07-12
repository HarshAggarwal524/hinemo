"""
HinEmo Data Collection — YouTube Comment Collector (FREE, primary source)
=============================================================================
Genuinely free: 10,000 quota units/day, no credit card, no approval queue.

Why this works well for us:
  - commentThreads.list costs only 1 unit per page of up to 100 comments.
  - videos.list / playlistItems.list cost 1 unit per call (up to 50 items).
  - Only search.list is expensive (100 units/call, effectively ~100
    searches/day). So the strategy is: spend search sparingly to FIND
    videos, then spend almost the whole budget READING comments.
  - YouTube comment sections on Bollywood, cricket, political, and
    religious/motivational Indian content are some of the most naturally
    Hinglish text available anywhere online.

Strategy: search sparingly to build a candidate pool, batch-check real
comment counts via videos.list (nearly free), keep only the top N
highest-comment-count videos, THEN pull comments from those.

SETUP:
  pip install google-api-python-client python-dotenv
  1. Go to console.cloud.google.com -> create a project
  2. Enable "YouTube Data API v3"
  3. Create credentials -> API key (no OAuth needed for public read access)
  4. Put it in .env as YOUTUBE_API_KEY=your_key
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

RAW_TARGET_PER_EMOTION = 11000
COMMENTS_PER_VIDEO_PAGE = 100          # max allowed by the API             
SEARCHES_USED = 0
MAX_COMMENTS_PER_VIDEO = RAW_TARGET_PER_EMOTION // 6   # no single video supplies more than ~1/6 of a class
MAX_SEARCHES_PER_RUN = 15              # stay well under the ~100/day search cap

# ---------------------------------------------------------------------------
# TOPIC MAP — search queries used SPARINGLY (just to discover video IDs)
# ---------------------------------------------------------------------------
EMOTION_SEARCH_QUERIES = {
    "anger": ["farmer protest India speech", "cricket loss reaction India",
              "traffic rant India vlog", "political debate India angry"],
    "happiness": ["diwali celebration vlog India", "cricket win reaction India",
                  "Indian wedding vlog", "Bollywood song reaction happy"],
    "sadness": ["emotional Indian family vlog", "sad Bollywood scene reaction",
                "India farmer struggle documentary", "breakup story Hindi"],
    "fear": ["India crime news report", "horror story Hindi vlog",
             "health scare India news", "exam stress India vlog"],
    "surprise": ["shocking news India", "unexpected result India reaction",
                 "viral video India reaction", "plot twist Bollywood reaction"],
    "disgust": ["food hygiene India news", "corruption news India reaction",
                "gross prank India vlog", "public cleanliness India complaint"],
}

# Manually curated video IDs — skips search.list entirely for these.
SEED_VIDEO_IDS = {
    # "anger": ["dQw4w9WgXcQ", ...],   # fill in as you find good sources
}


def discover_videos_for_emotion(emotion, queries, max_videos=40):
    """Uses search.list SPARINGLY. Pulls a wider candidate pool than we'll
    actually use, since rank_and_filter_videos() cuts this down later."""
    global SEARCHES_USED
    video_ids = list(SEED_VIDEO_IDS.get(emotion, []))

    for query in queries:
        if SEARCHES_USED >= MAX_SEARCHES_PER_RUN or len(video_ids) >= max_videos:
            break
        try:
            resp = youtube.search().list(
                q=query, part="id", type="video",
                maxResults=25, relevanceLanguage="hi",
                regionCode="IN",
            ).execute()
            SEARCHES_USED += 1
            for item in resp.get("items", []):
                vid = item["id"].get("videoId")
                if vid and vid not in video_ids:
                    video_ids.append(vid)
        except HttpError as e:
            print(f"  [!] search failed for '{query}': {e}")
            time.sleep(2)

    return video_ids[:max_videos]


def check_comment_counts(video_ids):
    """Batch-checks comment counts for up to 50 video IDs per call (1 quota
    unit total). Returns a dict: {video_id: comment_count}"""
    counts = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            resp = youtube.videos().list(
                part="statistics",
                id=",".join(batch),
            ).execute()
            for item in resp.get("items", []):
                stats = item.get("statistics", {})
                counts[item["id"]] = int(stats.get("commentCount", 0))
        except HttpError as e:
            print(f"  [!] stats check failed for batch: {e}")
    return counts


def rank_and_filter_videos(video_ids, min_comments=100, keep_top_n=30):
    """Checks real comment counts and returns only the most promising
    videos — highest comment count first, dropping anything under
    min_comments."""
    counts = check_comment_counts(video_ids)
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    filtered = [(vid, c) for vid, c in ranked if c >= min_comments]

    print(f"    stats check: {len(video_ids)} candidates -> "
          f"{len(filtered)} above {min_comments} comments, "
          f"keeping top {min(keep_top_n, len(filtered))}")
    if filtered:
        top_counts = [c for _, c in filtered[:keep_top_n]]
        print(f"    comment counts of kept videos: {top_counts}")

    return [vid for vid, _ in filtered[:keep_top_n]]


def fetch_comments_for_video(video_id, emotion, max_comments):
    """Pulls comments up to max_comments, which is already capped by the
    caller at min(remaining budget, MAX_COMMENTS_PER_VIDEO) — so one huge
    video can't dominate a class."""
    rows = []
    page_token = None

    while len(rows) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=COMMENTS_PER_VIDEO_PAGE,
                pageToken=page_token,
                textFormat="plainText",
                order="relevance",
            ).execute()
        except HttpError as e:
            if "commentsDisabled" not in str(e):
                print(f"    [!] {video_id}: {e}")
            break

        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "source_id": item["id"],
                "source": "youtube_comment",
                "video_id": video_id,
                "target_emotion": emotion,
                "text": snippet.get("textDisplay", "").strip(),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt"),
            })
            if len(rows) >= max_comments:
                break

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)

    return rows


def main():
    all_rows = []
    for emotion, queries in EMOTION_SEARCH_QUERIES.items():
        print(f"\n=== {emotion} ===")
        candidate_ids = discover_videos_for_emotion(emotion, queries)
        print(f"  discovered {len(candidate_ids)} raw candidates")

        video_ids = rank_and_filter_videos(candidate_ids, min_comments=100, keep_top_n=30)

        emotion_rows = []
        for vid in video_ids:
            remaining = RAW_TARGET_PER_EMOTION - len(emotion_rows)
            if remaining <= 0:
                break
            per_video_cap = min(remaining, MAX_COMMENTS_PER_VIDEO)
            rows = fetch_comments_for_video(vid, emotion, max_comments=per_video_cap)
            emotion_rows.extend(rows)
            print(f"    {vid}: +{len(rows)} comments (running total: {len(emotion_rows)})")

        all_rows.extend(emotion_rows)
        if len(emotion_rows) < RAW_TARGET_PER_EMOTION:
            print(f"  [!] {emotion} short of target ({len(emotion_rows)}/{RAW_TARGET_PER_EMOTION}). "
                  f"Add more entries to SEED_VIDEO_IDS['{emotion}'] and rerun — "
                  f"this costs 0 search-quota since you're supplying video IDs directly.")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"])
    df.to_csv("youtube_raw.csv", index=False)
    print(f"\nSaved {len(df)} unique comments to youtube_raw.csv")
    print(f"Search quota used this run: {SEARCHES_USED}/{MAX_SEARCHES_PER_RUN}")
    print("Feed this through 3_label_emotions_gpt.py next — 'target_emotion' is")
    print("still just a collection hint, not a gold label.")


if __name__ == "__main__":
    main()