#!/usr/bin/env python3
"""Attach ground-truth answers from a HotpotQA queries.jsonl file to result JSON outputs.

The script reads IDs from the query file and injects a `ground_truth` field into each
result item whose `id` matches the query record.

Example:
    python scripts/add_ground_truth.py \
        --queries /tmp/masoo20/inception_rag/datasets/hotpotqa/queries.jsonl \
        --files /tmp/masoo20/inception_rag/results/Qwen-Qwen2.5-3B-Instruct/hotpotqa_attack_100_with_ad.json \
               /tmp/masoo20/inception_rag/results/Qwen-Qwen2.5-3B-Instruct/hotpotqa_attack_100.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Union


def load_ground_truths(queries_path: str) -> Dict[str, str]:
    """Map query ID -> gold answer from the HotpotQA queries.jsonl file."""
    mapping: Dict[str, str] = {}

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            item_id = record.get("_id") or record.get("id")
            metadata = record.get("metadata") or {}
            answer = metadata.get("answer")

            if item_id is not None and answer is not None:
                mapping[str(item_id)] = str(answer)

    return mapping


def attach_ground_truths(payload: Union[dict, list], gt_map: Dict[str, str]) -> int:
    """Mutate payload in-place and return the number of records updated."""
    updated = 0

    def update_item(item: dict):
        nonlocal updated
        if not isinstance(item, dict):
            return

        item_id = item.get("id") or item.get("_id") or item.get("question_id")
        if item_id is None:
            return

        key = str(item_id)
        if key in gt_map:
            item["ground_truth"] = gt_map[key]
            updated += 1

    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                update_item(item)
            return updated

        if isinstance(payload, list):
            for item in payload:
                update_item(item)
            return updated

    if isinstance(payload, list):
        for item in payload:
            update_item(item)

    return updated


def process_file(file_path: str, gt_map: Dict[str, str]) -> int:
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    updated = attach_ground_truths(payload, gt_map)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Updated {path}: {updated} records with ground_truth")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Add ground_truth answers from queries.jsonl into attack result JSON files.")
    parser.add_argument("--queries", type=str, required=True, help="Path to queries.jsonl")
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="One or more result JSON files to update (e.g. hotpotqa_attack_100_with_ad.json hotpotqa_attack_100.json)",
    )
    args = parser.parse_args()

    gt_map = load_ground_truths(args.queries)
    total = 0
    for file_path in args.files:
        total += process_file(file_path, gt_map)

    print(f"Done. Updated {total} total records across {len(args.files)} files.")


if __name__ == "__main__":
    main()
