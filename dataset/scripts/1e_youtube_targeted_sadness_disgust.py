"""
HinEmo Data Collection — Targeted Round 2: Sadness & Disgust
===================================================================
Round 1's topic-targeted collection had a poor emotion-hit-rate for
sadness/disgust specifically (broad categories like "sad movie scenes"
or "disgusting behavior" pulled mixed-emotion comment sections). This
round uses hand-picked videos tightened to death/tribute content (sadness)
and moral-outrage/exposé content (disgust), which should correlate much
more tightly with the target emotion.

Reuses the same water-filling allocation logic as 1d (fair distribution
across videos of very different sizes), but tags target_emotion explicitly
per video set, since these picks ARE topic-targeted this time.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
API_KEY = os.environ.get("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

COMMENTS_PER_PAGE = 100
TARGET_PER_EMOTION = 15000  # raw target per emotion for this round

EXISTING_RAW_PATHS = [
    "data/raw/youtube_raw.csv",
    "data/raw/youtube_generalized_raw.csv",
]
OUTPUT_PATH = "data/raw/youtube_targeted_round2.csv"

# Paste your checked video IDs here
SADNESS_VIDEO_IDS = [
    "YhaG5GcCvA8",
    "TZELC5o94ck",
    "BAYixBHQM70",
    "RemShT6JAHw",
    "iT9fD9-_z-E",
    "TGHqBX1YJRc",
    "as55YImkgoY",
    "9CTtIIUBc0k",
    "grGvQl8RZkM",
    "zyGwb84l80I",
    "EKYaQ0Xtl7I"
        
]

DISGUST_VIDEO_IDS = [
    "vjp8LjrJp3Y",
    "9eorW2IdK8M",
    "PCc_9eLIa_E",
    "Gv4tkDTeV3o",
    "TYTIY7azQ4s",
    "oeH0fOSJuX8",
    "ua-6XN4vdQ8"
    
]


def load_existing_ids():
    ids = set()
    for path in EXISTING_RAW_PATHS + [OUTPUT_PATH]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "source_id" in df.columns:
                ids.update(df["source_id"].astype(str))
    print(f"Loaded {len(ids)} existing comment IDs to avoid re-collecting")
    return ids


def check_comment_counts(video_ids):
    counts = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            resp = youtube.videos().list(part="statistics", id=",".join(batch)).execute()
            for item in resp.get("items", []):
                stats = item.get("statistics", {})
                counts[item["id"]] = int(stats.get("commentCount", 0))
        except HttpError as e:
            print(f"  [!] stats check failed: {e}")
    return counts


def water_filling_allocation(counts, target_total):
    remaining = dict(counts)
    allocation = {vid: 0 for vid in counts}
    remaining_target = target_total

    while remaining_target > 0 and remaining:
        share = remaining_target / len(remaining)
        fully_drained = {vid: c for vid, c in remaining.items() if c <= share}

        if not fully_drained:
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

    return allocation


def fetch_comments_for_video(video_id, existing_ids, max_comments, emotion_tag):
    rows = []
    page_token = None

    while len(rows) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=COMMENTS_PER_PAGE,
                pageToken=page_token, textFormat="plainText", order="time",
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
                "target_emotion": emotion_tag,
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


def collect_for_category(video_ids, emotion_tag, existing_ids):
    print(f"\n=== {emotion_tag} ===")
    counts = check_comment_counts(video_ids)
    for vid, c in counts.items():
        print(f"  {vid}: {c} comments available")

    allocation = water_filling_allocation(counts, TARGET_PER_EMOTION)

    all_rows = []
    for vid, target in allocation.items():
        if target <= 0:
            continue
        rows = fetch_comments_for_video(vid, existing_ids, target, emotion_tag)
        all_rows.extend(rows)
        print(f"  {vid}: collected {len(rows)} (target was {target})")

    return all_rows


def main():
    existing_ids = load_existing_ids()

    all_rows = []
    all_rows.extend(collect_for_category(SADNESS_VIDEO_IDS, "sadness", existing_ids))
    all_rows.extend(collect_for_category(DISGUST_VIDEO_IDS, "disgust", existing_ids))

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


if __name__ == "__main__":
    main()