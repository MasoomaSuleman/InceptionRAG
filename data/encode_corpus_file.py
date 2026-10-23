"""Encode a JSONL corpus file with the Contriever encoder.

Example:
    python data/encode_corpus_file.py \
        --corpus data/datasets/hotpotqa/corpus_10200.jsonl \
        --output-dir data/corpus \
        --name hotpotqa_10200
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from download_datasets import encode_corpus_contriever


def load_corpus(corpus_path: Path) -> dict[str, dict[str, Any]]:
    """Load BEIR-style JSONL records keyed by corpus ID."""
    corpus: dict[str, dict[str, Any]] = {}
    with corpus_path.open("r", encoding="utf-8-sig") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                corpus_id = str(record["_id"])
            except (KeyError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid corpus record at {corpus_path}:{line_number}"
                ) from error
            if corpus_id in corpus:
                raise ValueError(
                    f"Duplicate corpus ID {corpus_id!r} at "
                    f"{corpus_path}:{line_number}"
                )
            corpus[corpus_id] = record
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode a JSONL corpus file with Contriever."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="corpus_10200")
    parser.add_argument("--max-corpus", type=int)
    args = parser.parse_args()

    if args.max_corpus is not None and args.max_corpus < 1:
        parser.error("--max-corpus must be positive")

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    corpus = load_corpus(args.corpus)
    logging.info("Loaded %d corpus records from %s", len(corpus), args.corpus)
    encode_corpus_contriever(
        corpus,
        args.output_dir,
        args.name,
        max_docs=args.max_corpus,
    )


if __name__ == "__main__":
    main()
