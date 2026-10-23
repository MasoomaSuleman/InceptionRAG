"""Download Wikipedia article text for corpus IDs in an example JSON file.

The corpus IDs are Wikipedia page IDs. The script uses Wikipedia's API
plaintext extract endpoint instead of scraping HTML.

Example:
    python data/download_wikipedia_corpus.py \
        --examples data/modified_example/hotpotqa_100_with_ad_with_ids.json \
        --output-dir data/modified_example/hotpotqa/corpus

Each article is saved as:
    <output-dir>/<corpus-id>/text.txt
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import requests


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "InceptionRAG corpus downloader/1.0"


def load_corpus_ids(examples_path: Path) -> list[str]:
    """Load unique corpus IDs in their first-seen order."""
    with examples_path.open("r", encoding="utf-8") as examples_file:
        examples: list[dict[str, Any]] = json.load(examples_file)

    corpus_ids: list[str] = []
    seen: set[str] = set()
    for example in examples:
        for corpus_id in example.get("corpus_ids", []):
            normalized_id = str(corpus_id).strip()
            if not normalized_id:
                continue
            if normalized_id not in seen:
                seen.add(normalized_id)
                corpus_ids.append(normalized_id)

    return corpus_ids


def download_article(
    session: requests.Session,
    corpus_id: str,
    output_dir: Path,
    retries: int,
    overwrite: bool,
) -> None:
    """Download one Wikipedia plaintext extract and save it to disk."""
    if not corpus_id.isdigit():
        raise ValueError(f"Corpus ID is not numeric: {corpus_id!r}")

    article_dir = output_dir / corpus_id
    text_path = article_dir / "text.txt"
    if text_path.exists() and not overwrite:
        return

    response = None
    for attempt in range(retries + 1):
        response = session.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "pageids": corpus_id,
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
            },
            timeout=30,
        )
        if response.ok:
            break
        if attempt < retries:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait_seconds = float(retry_after)
                except ValueError:
                    wait_seconds = 2**attempt
            else:
                wait_seconds = 2**attempt
            time.sleep(wait_seconds + random.uniform(0, 1))

    assert response is not None
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise ValueError(f"Wikipedia page not found for corpus ID {corpus_id}")

    text = pages[0].get("extract", "").strip()
    if not text:
        raise ValueError(f"Wikipedia page has no text for corpus ID {corpus_id}")

    article_dir.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Wikipedia text for corpus IDs in example JSON."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests; defaults to 2.0 to avoid rate limits.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        help="Download only the first N unique corpus IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download articles even when text.txt already exists.",
    )
    args = parser.parse_args()

    if args.delay < 0 or args.retries < 0:
        parser.error("--delay and --retries must be non-negative")

    corpus_ids = load_corpus_ids(args.examples)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        corpus_ids = corpus_ids[: args.limit]

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    downloaded = 0
    skipped = 0
    failures: dict[str, str] = {}

    try:
        for index, corpus_id in enumerate(corpus_ids, start=1):
            text_path = args.output_dir / corpus_id / "text.txt"
            if text_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[{index}/{len(corpus_ids)}] exists: {corpus_id}")
                continue

            try:
                download_article(
                    session,
                    corpus_id,
                    args.output_dir,
                    args.retries,
                    args.overwrite,
                )
            except (requests.RequestException, ValueError) as error:
                failures[corpus_id] = str(error)
                print(f"[{index}/{len(corpus_ids)}] failed: {corpus_id}: {error}")
            else:
                downloaded += 1
                print(f"[{index}/{len(corpus_ids)}] downloaded: {corpus_id}")

            if index < len(corpus_ids):
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed files are preserved.")
    finally:
        failure_path = args.output_dir / "download_errors.json"
        if failures:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    print(
        f"Completed: {downloaded} downloaded, {skipped} skipped, "
        f"{len(failures)} failed out of {len(corpus_ids)} corpus IDs."
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
