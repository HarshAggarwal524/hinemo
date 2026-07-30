"""
HinEmo Data Collection — Generalized YouTube Collector (water-filled allocation)
====================================================================================
No topic targeting — every comment gets target_emotion left blank, since
GPT labeling in 3_label_emotions_gpt.py determines the real label regardless
of source.

Allocation strategy ("water-filling"): given a total target (default 50,000)
spread across however many videos you hand-pick, this repeatedly:
  1. Divides the remaining target evenly across remaining videos
  2. Fully drains any video whose available comment count is BELOW that
     even share (since it can't supply more than it has anyway)
  3. Removes drained videos from the pool, recomputes the share for what's
     left, and repeats
  4. Once every remaining video has more comments than the current share,
     splits the rest evenly across them

This means small videos get fully used, and the remaining budget
concentrates fairly on the videos that can actually support it — rather
than either wasting quota trying to over-pull small videos, or letting one
huge video dominate the batch.

Checks against your existing collected files by source_id, so re-picking a
video you already have comments from won't create duplicates or waste quota.

SETUP: needs YOUTUBE_API_KEY in .env (same as 1c).
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

TOTAL_TARGET = 50000
COMMENTS_PER_PAGE = 100

EXISTING_RAW_PATHS = ["youtube_raw.csv", "youtube_generalized_raw.csv"]
OUTPUT_PATH = "youtube_generalized_raw.csv"

# ---------------------------------------------------------------------------
# Paste video IDs here as you find them (the part after v= in the URL).
# No emotion labeling needed — pick videos with lots of varied comments.
# ---------------------------------------------------------------------------
MANUAL_VIDEO_IDS = [
    "nx-mGN2Fz5M",
    "1tY5ZW6FNBI",
    "acunHsQcHS0",
    "6fzqjBN9LBw",
    "BGwendtSifY",
    "lb7gZnyZH0c",
    "M2Iz17NyWxE",
    "1CH8nUaA54c",
    "57ypmQtyvSQ",
    "7AuRTkNKJGQ",   
    "7s3sn4ph_uM",
    "IKgD3VsV-h8",
    "lPA8lq9Sdwc"
]



def load_existing_ids():
    ids = set()
    for path in EXISTING_RAW_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "source_id" in df.columns:
                ids.update(df["source_id"].astype(str))
                print(f"Loaded {len(df)} existing comment IDs from {path}")
    return ids


def check_comment_counts(video_ids):
    """Batch-checks real comment counts (1 quota unit per 50 IDs)."""
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


def water_filling_allocation(counts, target_total):
    """
    Distributes target_total across videos as evenly as possible, fully
    draining any video below its current fair share before recomputing
    the share for the rest. Returns {video_id: comments_to_pull}.
    """
    remaining = dict(counts)          # video_id -> available comments
    allocation = {vid: 0 for vid in counts}
    remaining_target = target_total

    while remaining_target > 0 and remaining:
        share = remaining_target / len(remaining)
        fully_drained = {vid: c for vid, c in remaining.items() if c <= share}

        if not fully_drained:
            # every remaining video has more than the current share —
            # split what's left evenly (with leftover units going to the
            # first few videos to land on an exact integer total)
            n = len(remaining)
            per_video = remaining_target // n
            leftover = remaining_target - per_video * n
            for i, vid in enumerate(remaining):
                take = per_video + (1 if i < leftover else 0)
                allocation[vid] += take
            remaining_target = 0
            break

        for vid, c in fully_drained.items():
            allocation[vid] += c
            remaining_target -= c
            del remaining[vid]
        # loop again: share gets recomputed on the shrunk pool + reduced target

    return allocation


def fetch_comments_for_video(video_id, existing_ids, max_comments):
    rows = []
    page_token = None

    while len(rows) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=COMMENTS_PER_PAGE,
                pageToken=page_token,
                textFormat="plainText",
                order="time",   # 'relevance' is documented as non-deterministic
                                 # and truncates pagination early — use 'time'
                                 # for complete, reliable retrieval
            ).execute()
        except HttpError as e:
            if "commentsDisabled" not in str(e):
                print(f"    [!] {video_id}: {e}")
            break

        for item in resp.get("items", []):
            comment_id = item["id"]
            if comment_id in existing_ids:
                continue
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "source_id": comment_id,
                "source": "youtube_comment",
                "video_id": video_id,
                "target_emotion": "",
                "text": snippet.get("textDisplay", "").strip(),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt"),
            })
            existing_ids.add(comment_id)
            if len(rows) >= max_comments:
                break

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)

    return rows


def main():
    if not MANUAL_VIDEO_IDS:
        print("MANUAL_VIDEO_IDS is empty — add some video IDs before running.")
        return

    print(f"Checking real comment counts for {len(MANUAL_VIDEO_IDS)} videos...")
    counts = check_comment_counts(MANUAL_VIDEO_IDS)
    for vid, c in counts.items():
        print(f"  {vid}: {c} comments available (includes replies — actual "
              f"top-level pullable count may be lower)")

    existing_ids = load_existing_ids()
    all_rows = []
    actually_pulled = {}   # video_id -> how many we really got, for reallocation

    remaining_videos = dict(counts)
    attempt = 1
    max_attempts = 4   # cap retries so we don't loop forever if videos are just tapped out

    while len(all_rows) < TOTAL_TARGET and remaining_videos and attempt <= max_attempts:
        gap = TOTAL_TARGET - len(all_rows)
        print(f"\n--- Allocation pass {attempt}: need {gap} more, "
              f"{len(remaining_videos)} videos still have room ---")

        allocation = water_filling_allocation(remaining_videos, gap)
        for vid, take in sorted(allocation.items(), key=lambda x: -x[1]):
            print(f"  {vid}: pull {take}")

        for vid, target in allocation.items():
            if target <= 0:
                continue
            rows = fetch_comments_for_video(vid, existing_ids, max_comments=target)
            all_rows.extend(rows)
            actually_pulled[vid] = actually_pulled.get(vid, 0) + len(rows)
            print(f"  {vid}: got {len(rows)} (running total: {len(all_rows)})")

            # if a video delivered less than asked, it's tapped out — drop it
            # from the pool so the next pass doesn't retry it
            if len(rows) < target:
                remaining_videos.pop(vid, None)
            else:
                remaining_videos[vid] = max(remaining_videos.get(vid, 0) - len(rows), 0)

        attempt += 1

    if len(all_rows) < TOTAL_TARGET:
        print(f"\n[!] Still short: {len(all_rows)}/{TOTAL_TARGET}. "
              f"These videos are genuinely tapped out — add more video IDs "
              f"to MANUAL_VIDEO_IDS and rerun to close the gap.")

    if not all_rows:
        print("No new comments collected.")
        return

    new_df = pd.DataFrame(all_rows)

    if os.path.exists(OUTPUT_PATH):
        prior = pd.read_csv(OUTPUT_PATH)
        combined = pd.concat([prior, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["source_id"])
    else:
        combined = new_df

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} total comments to {OUTPUT_PATH} "
          f"({len(new_df)} newly added this run)")
    print("Merge this with youtube_raw.csv before running 3_label_emotions_gpt.py")


if __name__ == "__main__":
    main()