"""
Data extraction module for AppleSupport conversations from the Customer Support on Twitter dataset.
Efficiently streams raw CSV or ZIP archive without exhausting memory.
"""

import os
import csv
import io
import zipfile
import html
from typing import Generator, Dict, Any, Optional
from pathlib import Path


DEFAULT_ARCHIVE_PATH = r"C:\Users\Prava\Downloads\archive.zip"
TARGET_AUTHOR = "AppleSupport"
TARGET_HANDLE = "@AppleSupport"


def clean_tweet_text(text: str) -> str:
    """Unescape HTML entities and strip superfluous whitespace."""
    if not text:
        return ""
    cleaned = html.unescape(text)
    # Normalize unicode spaces and clean up
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    return cleaned.strip()


def stream_raw_tweets(
    source_path: str,
    max_rows: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream rows from twcs.csv directly, either from inside a .zip file or a raw .csv file.
    
    Yields dictionary representing raw tweet record.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {source_path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            # Look for twcs.csv or twcs/twcs.csv
            names = z.namelist()
            csv_name = next(
                (n for n in names if n.lower().endswith("twcs.csv")), None
            )
            if not csv_name:
                raise ValueError(
                    f"No twcs.csv found inside {path}. Archive contains: {names}"
                )
            with z.open(csv_name) as f:
                text_stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text_stream)
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break
                    yield row
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                yield row
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .zip or .csv")


def stream_applesupport_tweets(
    source_path: str = DEFAULT_ARCHIVE_PATH,
    max_scan_rows: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Filter and stream only tweets relevant to AppleSupport (authored by or mentioning @AppleSupport).
    """
    for row in stream_raw_tweets(source_path, max_rows=max_scan_rows):
        author = row.get("author_id", "")
        text = row.get("text", "")
        
        is_apple_agent = author == TARGET_AUTHOR
        mentions_apple = TARGET_HANDLE.lower() in text.lower()
        
        if is_apple_agent or mentions_apple:
            clean_text = clean_tweet_text(text)
            yield {
                "tweet_id": row.get("tweet_id", "").strip(),
                "author_id": author.strip(),
                "inbound": row.get("inbound", "").strip().lower() == "true",
                "created_at": row.get("created_at", "").strip(),
                "text": clean_text,
                "response_tweet_id": row.get("response_tweet_id", "").strip(),
                "in_response_to_tweet_id": row.get("in_response_to_tweet_id", "").strip(),
            }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Extract AppleSupport tweets from Twitter dataset.")
    parser.add_argument("--source", type=str, default=DEFAULT_ARCHIVE_PATH, help="Path to archive.zip or twcs.csv")
    parser.add_argument("--max-rows", type=int, default=100000, help="Max raw rows to scan (default 100k)")
    parser.add_argument("--out", type=str, default="data/sample/applesupport_raw_sample.jsonl", help="Output path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning from: {args.source} (scanning first {args.max_rows} rows)...")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for item in stream_applesupport_tweets(args.source, max_scan_rows=args.max_rows):
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"Extracted {count} AppleSupport tweets to {out_path}")
