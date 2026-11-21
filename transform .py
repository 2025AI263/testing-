"""Transform step: cleaning, topic inference and toxicity scoring for the pipeline."""
import html
import os
import re

try:
    import config
except ImportError:  # keeps infer_topic usable on its own
    config = None

DEFAULT_TOPIC_KEYWORDS = {
    # order matters: the first topic with a matching keyword wins
    "tech":      ["iphone", "android", "review", "unboxing", "tesla", "ev", "phone", "laptop",
                  "camera", "ai", "tech", "pixel", "galaxy", "macbook", "samsung", "xiaomi",
                  "fold", "apple", "google", "headphones", "earbuds", "chip"],
    "challenge": ["challenge", "survive", "last to", "$1", "$1,000", "million", "give away", "giveaway"],
    "music":     ["official video", "music video", "lyrics", "song", "album", "live", "concert",
                  "never gonna give you up", "remix", "audio"],
    "gaming":    ["minecraft", "fortnite", "gameplay", "gaming", "playthrough", "roblox"],
    "education": ["how to", "explained", "tutorial", "learn", "science", "history"],
    "vlog":      ["vlog", "day in", "behind the scenes", "q&a", "my life"],
    "news":      ["news", "election", "breaking", "update", "announcement"],
}
# config.TOPIC_KEYWORDS (if you defined one) wins; otherwise use the defaults above.
TOPIC_KEYWORDS = getattr(config, "TOPIC_KEYWORDS", None) or DEFAULT_TOPIC_KEYWORDS


def infer_topic(title: str) -> str:
    """Assign a broad topic from whole words / phrases in a video title."""
    title = str(title or "").lower().strip()
    if not title:
        return "general"
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            keyword = keyword.strip().lower()
            if not keyword:
                continue
            if re.fullmatch(r"[^a-z0-9]+", keyword):
                if keyword in title:
                    return topic
                continue
            # \b-style guard that also works next to symbols such as "$1"
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", title):
                return topic
    return "general"


_MARKUP = re.compile(r"</?(?:a|br|b|i|u|span)\b[^>]*>", re.I)


def clean_text(value) -> str:
    """Unescape HTML entities, drop simple markup, collapse whitespace."""
    text = html.unescape("" if value is None else str(value))
    text = _MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def run():
    """Read raw JSONL -> cleaned, topic-tagged, scored DataFrame (returned, not saved)."""
    import pandas as pd
    import toxicity

    if not os.path.exists(config.RAW_PATH):
        raise FileNotFoundError(f"{config.RAW_PATH} not found. Run `python main.py extract` first.")

    df = pd.read_json(config.RAW_PATH, lines=True, dtype=False, convert_dates=False)
    raw_rows = len(df)
    df = df.dropna(subset=["comment_id", "video_id"]).drop_duplicates("comment_id")

    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() > 0].copy()
    df["text_len"] = df["text_clean"].str.len()
    df["topic"] = df["video_title"].map(infer_topic) if "video_title" in df.columns else "general"

    results = toxicity.predict_many(df["text_clean"].tolist())
    df["toxicity_score"] = [r["toxicity"] for r in results]
    df["is_toxic"] = [int(r["is_toxic"]) for r in results]
    cats = pd.DataFrame([r["categories"] for r in results], index=df.index).fillna(0.0)
    cats.columns = [f"cat_{c.replace(' ', '_')}" for c in cats.columns]
    df = pd.concat([df, cats], axis=1)

    if not getattr(config, "KEEP_AUTHORS", False):
        df = df.drop(columns=["author"], errors="ignore")   # commenter handles are personal data

    print(f"TRANSFORM: {raw_rows} raw rows -> {len(df)} comments "
          f"({int(df['is_toxic'].sum())} flagged toxic, backend={config.TOXICITY_BACKEND})")
    print("Topics:", df["topic"].value_counts().to_dict())
    return df.reset_index(drop=True)


if __name__ == "__main__":
    run()
