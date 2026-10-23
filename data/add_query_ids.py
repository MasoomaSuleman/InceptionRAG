"""Add BEIR query IDs, answers, and corpus IDs to an example JSON file.

Example:
    python data/add_query_ids.py \
        --examples data/example/hotpotqa_100_with_ad.json \
        --queries data/datasets/hotpotqa/queries.jsonl \
        --qrels data/datasets/hotpotqa/qrels/train.tsv \
        data/datasets/hotpotqa/qrels/test.tsv \
        --output data/modified_example/hotpotqa_100_with_ad_with_ids.json

Use --in-place instead of --output to replace the examples file.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_queries(queries_path: Path) -> dict[str, list[dict[str, str]]]:
    """Return a mapping from query text to matching BEIR records."""
    queries_by_text: dict[str, list[dict[str, str]]] = {}

    with queries_path.open("r", encoding="utf-8") as queries_file:
        for line_number, line in enumerate(queries_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            query_text = record["text"]
            query = {
                "query_id": record["_id"],
                "true_answer": record["metadata"]["answer"],
            }

            matching_queries = queries_by_text.setdefault(query_text, [])
            if query not in matching_queries:
                matching_queries.append(query)

    return queries_by_text


def load_corpus_ids(qrels_paths: list[Path]) -> dict[str, list[str]]:
    """Return relevant corpus IDs grouped by query ID from qrels files."""
    corpus_ids_by_query: dict[str, list[str]] = {}

    for qrels_path in qrels_paths:
        with qrels_path.open("r", encoding="utf-8") as qrels_file:
            header = next(qrels_file).rstrip("\n").split("\t")
            expected_header = ["query-id", "corpus-id", "score"]
            if header != expected_header:
                raise ValueError(
                    f"Unexpected qrels header in {qrels_path}: {header!r}; "
                    f"expected {expected_header!r}"
                )

            for line_number, line in enumerate(qrels_file, start=2):
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"Invalid qrels row at {qrels_path}:{line_number}: "
                        f"{line.rstrip()!r}"
                    )

                query_id, corpus_id, score = fields
                if int(score) <= 0:
                    continue
                ids_for_query = corpus_ids_by_query.setdefault(query_id, [])
                if corpus_id not in ids_for_query:
                    ids_for_query.append(corpus_id)

    return corpus_ids_by_query


def add_query_ids(
    examples_path: Path,
    queries_path: Path,
    qrels_paths: list[Path],
    output_path: Path,
) -> tuple[int, int, int]:
    """Add query metadata and corpus IDs and write the resulting JSON array."""
    with examples_path.open("r", encoding="utf-8") as examples_file:
        examples: list[dict[str, Any]] = json.load(examples_file)

    queries_by_text = load_queries(queries_path)
    corpus_ids_by_query = load_corpus_ids(qrels_paths)
    matched = 0
    unmatched_questions: list[str] = []
    ambiguous_questions: list[tuple[str, list[str]]] = []

    for example in examples:
        question = example["question"]
        matching_queries = queries_by_text.get(question, [])
        if not matching_queries:
            unmatched_questions.append(question)
            continue
        if len(matching_queries) > 1:
            ambiguous_questions.append((question, matching_queries))
            continue

        query_id = matching_queries[0]["query_id"]
        corpus_ids = corpus_ids_by_query.get(query_id)
        if not corpus_ids:
            raise ValueError(
                f"No relevant corpus IDs found for query ID {query_id!r} "
                f"({question!r})"
            )

        example.pop("_id", None)
        example.update(matching_queries[0])
        example["corpus_ids"] = corpus_ids
        matched += 1

    if unmatched_questions or ambiguous_questions:
        problem_lines = []
        if unmatched_questions:
            problem_lines.append(
                f"{len(unmatched_questions)} example question(s) did not match "
                "queries.jsonl:"
            )
            problem_lines.extend(
                f"  - {question}" for question in unmatched_questions[:5]
            )
        if ambiguous_questions:
            problem_lines.append(
                f"{len(ambiguous_questions)} example question(s) matched "
                "multiple query IDs:"
            )
            problem_lines.extend(
                f"  - {question!r}: {ids}"
                for question, ids in ambiguous_questions[:5]
            )
        problem_count = len(unmatched_questions) + len(ambiguous_questions)
        shown_count = min(problem_count, 5)
        if problem_count > shown_count:
            problem_lines.append(f"  ... and {problem_count - shown_count} more")
        preview = "\n".join(
            problem_lines
        )
        raise ValueError(preview)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(examples, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    return matched, len(queries_by_text), len(corpus_ids_by_query)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add query_id and true_answer values by matching question text."
        )
    )
    parser.add_argument(
        "--examples",
        type=Path,
        required=True,
        help="JSON array containing question fields.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="BEIR queries.jsonl file.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        nargs="+",
        required=True,
        help="One or more BEIR qrels TSV files.",
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output",
        type=Path,
        help="Path for the updated JSON file.",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the examples file after all questions match.",
    )
    args = parser.parse_args()

    output_path = args.examples if args.in_place else args.output
    assert output_path is not None
    matched, total_queries, total_qrels_queries = add_query_ids(
        args.examples,
        args.queries,
        args.qrels,
        output_path,
    )
    print(
        f"Added query IDs and true answers to {matched} example(s) using "
        f"{total_queries} query records and corpus IDs for "
        f"{total_qrels_queries} queries. "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
