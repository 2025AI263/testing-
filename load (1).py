"""LOAD: persist the transformed DataFrame to CSV + SQLite.

Schema: two normalized tables (videos 1-N comments), plus indexes for EDA.
The database is built in a temporary file and swapped in only if everything succeeded,
so a failed run can no longer destroy the previous database.
"""
import os, re, sqlite3
import pandas as pd
import config

DDL = """
CREATE TABLE videos (
    video_id      TEXT PRIMARY KEY,
    video_title   TEXT,
    channel       TEXT,
    published_at  TEXT,
    view_count    INTEGER,
    like_count    INTEGER,
    comment_count INTEGER
);
CREATE TABLE comments (
    comment_id     TEXT PRIMARY KEY,
    video_id       TEXT NOT NULL REFERENCES videos(video_id),
    text           TEXT,
    text_clean     TEXT,
    text_len       INTEGER,
    like_count     INTEGER,
    reply_count    INTEGER,
    published_at   TEXT,
    toxicity_score REAL,
    is_toxic       INTEGER,
    topic          TEXT
);
CREATE INDEX idx_comments_video ON comments(video_id);
CREATE INDEX idx_comments_toxic ON comments(is_toxic);
"""

# DataFrame column -> videos-table column. Comment-level like_count / published_at are NOT used here.
VIDEO_MAP = {
    "video_id": "video_id", "video_title": "video_title", "channel": "channel",
    "video_published_at": "published_at", "video_like_count": "like_count",
    "view_count": "view_count", "comment_count": "comment_count",
}
COMMENT_COLS = ["comment_id", "video_id", "text", "text_clean", "text_len", "like_count",
                "reply_count", "published_at", "toxicity_score", "is_toxic", "topic"]


def run(df: pd.DataFrame):
    if df is None or df.empty:
        raise ValueError("Nothing to load: the transformed DataFrame is empty.")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # ---- CSV (what the dashboard reads) ----
    df.to_csv(config.CLEAN_CSV, index=False)
    print(f"LOAD csv  -> {config.CLEAN_CSV} ({len(df)} rows)")

    # ---- SQLite (built aside, then swapped in) ----
    tmp_path = config.DB_PATH + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    con = sqlite3.connect(tmp_path)
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(DDL)

        cat_cols = [c for c in df.columns if c.startswith("cat_") and re.fullmatch(r"[a-z0-9_]+", c)]
        for c in cat_cols:                                   # keep per-category scores in the DB too
            con.execute(f'ALTER TABLE comments ADD COLUMN "{c}" REAL')

        videos = (df[[c for c in VIDEO_MAP if c in df.columns]]
                  .drop_duplicates("video_id").rename(columns=VIDEO_MAP))
        missing = [v for k, v in VIDEO_MAP.items() if k not in df.columns]
        if missing:
            print(f"LOAD note: no data for videos.{', '.join(missing)} (re-run extract to fill them).")
        videos.to_sql("videos", con, if_exists="append", index=False)

        ccols = [c for c in COMMENT_COLS if c in df.columns] + cat_cols
        df[ccols].to_sql("comments", con, if_exists="append", index=False)
        con.commit()
        print(pd.read_sql("SELECT COUNT(*) AS comments, SUM(is_toxic) AS toxic FROM comments",
                          con).to_string(index=False))
    except Exception:
        con.close()
        os.remove(tmp_path)
        raise
    con.close()
    os.replace(tmp_path, config.DB_PATH)
    print(f"LOAD db   -> {config.DB_PATH}")


if __name__ == "__main__":
    from transform import run as transform_run
    run(transform_run())
