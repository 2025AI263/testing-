"""EXTRACT: pull comments + video metadata from the YouTube Data API v3.

Output: data/raw_comments.jsonl  (one JSON object per comment, video stats joined in)
Quota: ~10,000 units/day. commentThreads.list / playlistItems.list / videos.list cost 1 unit
per call; resolving a channel by *name* with search.list costs 100 (handles like @mkbhd cost 1).

Video-level fields are prefixed (video_published_at, video_like_count) so they no longer
overwrite the comment-level like_count / published_at.
"""
import json, os, time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import config


class QuotaExceeded(RuntimeError):
    """The daily API quota is used up."""


class ApiError(RuntimeError):
    """API failure. The message never contains the request URL (which holds the API key)."""

    def __init__(self, status, reason):
        super().__init__(f"YouTube API error {status} ({reason or 'unknown reason'})")
        self.status, self.reason = status, reason


_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _reason(err):
    try:
        return json.loads(err.content.decode("utf-8"))["error"]["errors"][0]["reason"]
    except Exception:
        return ""


def _execute(request, retries=3):
    """Run an API request with retry/backoff; re-raise errors without leaking the key."""
    for attempt in range(retries + 1):
        try:
            return request.execute()
        except HttpError as e:
            status, reason = e.resp.status, _reason(e)
            if reason in _QUOTA_REASONS:
                raise QuotaExceeded("YouTube API daily quota exhausted") from None
            if (status in _RETRY_STATUS or reason == "rateLimitExceeded") and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ApiError(status, reason) from None


def get_youtube():
    if config.API_KEY in ("", "YOUR_API_KEY_HERE"):
        raise RuntimeError("Set YOUTUBE_API_KEY in your .env file (see .env.example)")
    return build("youtube", "v3", developerKey=config.API_KEY, cache_discovery=False)


def channel_uploads_playlist(youtube, channel):
    """Resolve a channel handle/ID/name to its uploads playlist ID."""
    channel = channel.strip()
    if channel.startswith("UC") and len(channel) == 24:
        params = {"id": channel}
    elif channel.startswith("@"):
        params = {"forHandle": channel}                       # 1 quota unit
    else:                                                     # plain name: 100 quota units
        resp = _execute(youtube.search().list(part="snippet", q=channel,
                                              type="channel", maxResults=1))
        if not resp.get("items"):
            raise ValueError(f"Channel not found: {channel}")
        params = {"id": resp["items"][0]["snippet"]["channelId"]}
    resp = _execute(youtube.channels().list(part="contentDetails", **params))
    items = resp.get("items") or []
    if not items:
        raise ValueError(f"Channel not found: {channel}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def videos_from_channel(youtube, playlist_id):
    """Most recent MAX_VIDEOS_PER_CHANNEL video IDs from an uploads playlist."""
    ids, token = [], None
    while len(ids) < config.MAX_VIDEOS_PER_CHANNEL:
        resp = _execute(youtube.playlistItems().list(
            part="contentDetails", playlistId=playlist_id, maxResults=50, pageToken=token))
        ids += [i["contentDetails"]["videoId"] for i in resp.get("items", [])]
        token = resp.get("nextPageToken")
        if not token:
            break
    return ids[:config.MAX_VIDEOS_PER_CHANNEL]


def video_stats(youtube, video_ids):
    """Map video_id -> video metadata (all keys prefixed so they cannot clash with comments)."""
    stats = {}
    for i in range(0, len(video_ids), 50):
        resp = _execute(youtube.videos().list(
            part="snippet,statistics", id=",".join(video_ids[i:i + 50])))
        for v in resp.get("items", []):
            vid, st = v["id"], v.get("statistics", {})
            stats[vid] = {
                "video_id": vid,
                "video_title": v["snippet"].get("title", ""),
                "channel": v["snippet"].get("channelTitle", ""),
                "video_published_at": v["snippet"].get("publishedAt", ""),
                "view_count": int(st.get("viewCount", 0)),
                "video_like_count": int(st.get("likeCount", 0)),
                "comment_count": int(st.get("commentCount", 0)),
            }
    return stats


def fetch_comments(youtube, video_id, cap):
    """Top-level comments for one video (paged), up to `cap`."""
    comments, token = [], None
    while len(comments) < cap:
        try:
            resp = _execute(youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                textFormat="plainText", pageToken=token))
        except ApiError as e:                       # comments disabled, video private, ...
            print(f"  [skip] {video_id}: {e}")
            break
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]
            s = top["snippet"]
            comments.append({
                "comment_id": top["id"],
                "video_id": video_id,
                "author": s.get("authorDisplayName", ""),
                "text": s.get("textDisplay", ""),
                "like_count": int(s.get("likeCount", 0)),
                "published_at": s.get("publishedAt", ""),
                "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
            })
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)                              # be polite with quota
    return comments[:cap]


def run():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    yt = get_youtube()
    tmp_path = config.RAW_PATH + ".tmp"              # never overwrite good data with a failed run
    n, complete = 0, True

    try:
        video_ids = list(config.VIDEO_IDS)
        for ch in config.CHANNELS:
            video_ids += videos_from_channel(yt, channel_uploads_playlist(yt, ch))
        video_ids = list(dict.fromkeys(video_ids))   # dedupe, keep order
        print(f"Target videos: {len(video_ids)}")

        stats = video_stats(yt, video_ids)
        with open(tmp_path, "w", encoding="utf-8") as out:
            for vid in video_ids:
                if vid not in stats:
                    print(f"  [skip] {vid}: no metadata (private or deleted)")
                    continue
                for c in fetch_comments(yt, vid, config.MAX_COMMENTS_PER_VIDEO):
                    out.write(json.dumps({**stats[vid], **c}, ensure_ascii=False) + "\n")
                    n += 1
                print(f"  {vid}: total comments so far = {n}")
    except QuotaExceeded as e:
        complete = False
        print(f"STOPPED: {e}. Keeping the {n} comments collected so far.")

    if n == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError("No comments collected; existing raw data was left untouched.")

    final = config.RAW_PATH if complete else config.RAW_PATH.replace(".jsonl", ".partial.jsonl")
    os.replace(tmp_path, final)
    print(f"EXTRACT {'done' if complete else 'PARTIAL'} -> {final} ({n} comments)")
    if not complete:
        print("Existing raw_comments.jsonl was not replaced. Re-run after the quota resets.")


if __name__ == "__main__":
    run()
