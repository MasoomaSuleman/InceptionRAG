"""Create a corpus JSONL subset from example-referenced corpus IDs.

The output contains every corpus record referenced by the examples plus the
requested number of additional corpus records. The large source JSONL file is
processed line by line.

Example:
    python data/create_corpus_subset.py \
        --examples data/modified_example/hotpotqa_100_with_ad_with_ids.json \
        --corpus data/datasets/hotpotqa/corpus.jsonl \
        --output data/datasets/hotpotqa/corpus_10200.jsonl \
        --additional-count 10000
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_referenced_ids(examples_path: Path) -> set[str]:
    """Load unique corpus IDs referenced by the example records."""
    with examples_path.open("r", encoding="utf-8-sig") as examples_file:
        examples: list[dict[str, Any]] = json.load(examples_file)

    referenced_ids: set[str] = set()
    for example in examples:
        referenced_ids.update(
            str(corpus_id) for corpus_id in example.get("corpus_ids", [])
        )
    return referenced_ids


def create_subset(
    examples_path: Path,
    corpus_path: Path,
    output_path: Path,
    additional_count: int,
) -> tuple[int, int]:
    """Write referenced records plus additional records to a new JSONL file."""
    referenced_ids = load_referenced_ids(examples_path)
    selected_additional_ids: set[str] = set()
    found_referenced_ids: set[str] = set()
    written_ids: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with corpus_path.open("r", encoding="utf-8-sig") as corpus_file, (
            temporary_output.open("w", encoding="utf-8")
        ) as output_file:
            for line_number, line in enumerate(corpus_file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {corpus_path}:{line_number}"
                    ) from error

                corpus_id = str(record["_id"])
                is_referenced = corpus_id in referenced_ids
                is_additional = (
                    not is_referenced
                    and corpus_id not in selected_additional_ids
                    and len(selected_additional_ids) < additional_count
                )

                if is_referenced:
                    found_referenced_ids.add(corpus_id)
                if is_additional:
                    selected_additional_ids.add(corpus_id)

                if (is_referenced or is_additional) and corpus_id not in written_ids:
                    output_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    written_ids.add(corpus_id)

        missing_ids = referenced_ids - found_referenced_ids
        if missing_ids:
            missing_preview = ", ".join(sorted(missing_ids)[:10])
            raise ValueError(
                f"{len(missing_ids)} referenced corpus ID(s) were not found "
                f"in {corpus_path}: {missing_preview}"
            )
        if len(selected_additional_ids) < additional_count:
            raise ValueError(
                f"Only found {len(selected_additional_ids)} additional corpus "
                f"records; requested {additional_count}"
            )

        temporary_output.replace(output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    return len(found_referenced_ids), len(selected_additional_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a corpus subset for the modified examples."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-count", type=int, default=10000)
    args = parser.parse_args()

    if args.additional_count < 0:
        parser.error("--additional-count must be non-negative")

    referenced_count, additional_count = create_subset(
        args.examples,
        args.corpus,
        args.output,
        args.additional_count,
    )
    print(
        f"Created {args.output} with {referenced_count} referenced and "
        f"{additional_count} additional corpus records "
        f"({referenced_count + additional_count} total)."
    )


if __name__ == "__main__":
    main()
