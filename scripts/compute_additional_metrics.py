from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


Pair = Tuple[str, str]  # (title, chunk id)

# Chunk categories tracked for every query:
#   attribute / bridge -> the two adversarial passages injected for the query
#   golden             -> the query's ground-truth passages
CATEGORIES = ("attribute", "bridge", "golden")

# Context keys in the pipeline output.
FIRST_K_KEY = "retrieval_first_k_context"
TOP_K_KEY = "retrieval_top_k_context"
RERANKER_KEY = "reranker_top_k_context"


def chunk_key(value: Any) -> str:
    """Normalize a chunk id so None / missing / 0 are handled consistently."""
    return "" if value is None else str(value)


def rate(numerator: int, denominator: int) -> Optional[float]:
    """numerator / denominator, or None when the denominator is 0 (undefined)."""
    return numerator / denominator if denominator else None


# ============================================================================
# Loading
# ============================================================================


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_adv_docs(path: str) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(f"Unsupported ad file format in {path}: expected a list of entries")


def load_corpus(path: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[Pair]]]:
    """Return (records grouped by title, golden (title, chunk) pairs grouped by query_id)."""
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported corpus file format in {path}: expected a JSON list")

    by_title: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    golden_by_query: Dict[str, Set[Pair]] = defaultdict(set)
    for record in payload:
        if not isinstance(record, dict):
            continue
        title = record.get("title")
        if title is None:
            continue
        by_title[str(title)].append(record)

        if str(record.get("doc_type", "")).lower() == "golden" and record.get("query_id") is not None:
            golden_by_query[str(record["query_id"])].add((str(title), chunk_key(record.get("chunk"))))
    return by_title, golden_by_query


# ============================================================================
# Retrieve data
# ============================================================================


def get_query_adv_passages(qid: str, adv_docs: Sequence[Dict]) -> Optional[List[Tuple[str, str, str]]]:
    """Return [(label, title, text), ...] for the query's adversarial passages."""
    query_adv_dict = next(
        (d for d in adv_docs if isinstance(d, dict) and str(d.get("id")) == str(qid)),
        None,
    )
    if not query_adv_dict:
        return None

    adv_text = query_adv_dict.get("ad_text")
    if not isinstance(adv_text, dict) or not adv_text:
        return None

    return [
        ("attribute", f"adv-{qid}-attribute", adv_text.get("attribute_document", "")),
        ("bridge", f"adv-{qid}-bridge", adv_text.get("bridge_document", "")),
    ]


def get_context(result: Dict, key: str) -> List:
    value = result.get(key)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def get_titles_chunk_pairs(entries: Sequence) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = next(
            (entry[k] for k in ("title", "doc_id", "id") if entry.get(k) is not None),
            None,
        )
        if title is None:
            continue
        pairs.add((str(title), chunk_key(entry.get("chunk"))))
    return pairs


# ============================================================================
# Pipeline stages
# ============================================================================


def build_stage_plan(reranker_enabled: bool) -> List[Tuple[str, str]]:
    """Ordered (stage name, context key) pairs for this pipeline.

    Chunker, no reranker:
        first_k_retrieval <- retrieval_first_k_context
        llm_context       <- retrieval_top_k_context      (top-k goes straight to the LLM)

    Chunker + reranker:
        first_k_retrieval <- retrieval_first_k_context
        top_k_retrieval   <- retrieval_top_k_context
        llm_context       <- reranker_top_k_context       (reranker output goes to the LLM)
    """
    if reranker_enabled:
        return [
            ("first_k_retrieval", FIRST_K_KEY),
            ("top_k_retrieval", TOP_K_KEY),
            ("llm_context", RERANKER_KEY),
        ]
    return [
        ("first_k_retrieval", FIRST_K_KEY),
        ("llm_context", TOP_K_KEY),
    ]


def stage_block(totals: Dict[str, int], retained: Dict[str, int], source_context: str) -> Dict[str, Any]:
    """Counts and rates for one pipeline stage (works for one query or pooled totals)."""
    block: Dict[str, Any] = {"source_context": source_context}
    for cat in CATEGORIES:
        block[f"{cat}_chunks_total"] = totals[cat]
        block[f"{cat}_chunks_retained"] = retained[cat]

    adv_total = totals["attribute"] + totals["bridge"]
    adv_retained = retained["attribute"] + retained["bridge"]
    block["adversarial_chunks_total"] = adv_total
    block["adversarial_chunks_retained"] = adv_retained

    block["attribute_chunk_retention_rate"] = rate(retained["attribute"], totals["attribute"])
    block["bridge_chunk_retention_rate"] = rate(retained["bridge"], totals["bridge"])
    # ACRR: (attribute + bridge chunks retained) / (attribute + bridge chunks total)
    block["adversarial_chunk_retention_rate"] = rate(adv_retained, adv_total)
    block["golden_chunk_retention_rate"] = rate(retained["golden"], totals["golden"])
    return block


# ============================================================================
# Query summaries
# ============================================================================


def build_query_summary(
    adv_passages: Sequence[Tuple[str, str, str]],
    golden_pairs: Set[Pair],
    result: Dict,
    corpus: Dict[str, List[Dict[str, Any]]],
    stage_plan: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    stage_pairs = {
        name: get_titles_chunk_pairs(get_context(result, key)) for name, key in stage_plan
    }

    # All chunks that exist in the corpus for each category of this query.
    category_pairs: Dict[str, Set[Pair]] = {"golden": set(golden_pairs)}

    for label, title, _text in adv_passages:
        records = corpus.get(title, [])
        category_pairs[label] = {
            (str(rec.get("title", "")), chunk_key(rec.get("chunk")))
            for rec in records
            if rec.get("title") is not None
        }

    totals = {cat: len(category_pairs[cat]) for cat in CATEGORIES}
    retained = {
        stage: {cat: len(category_pairs[cat] & pairs) for cat in CATEGORIES}
        for stage, pairs in stage_pairs.items()
    }

    return {
        "question": result.get("question", ""),
        "totals": totals,
        "retained": retained,
    }


# ============================================================================
# Main computation
# ============================================================================


def compute_metrics(adv_path: str, corpus_path: str, results_path: str) -> Dict[str, Any]:
    adv_docs = load_adv_docs(adv_path)
    corpus, golden_by_query = load_corpus(corpus_path)
    output = load_json(results_path)

    output_dict = output if isinstance(output, dict) else {}
    results = [r for r in output_dict.get("results", []) if isinstance(r, dict)]

    config = output_dict.get("config_yaml") or {}
    attack_changes = config.get("attack_changes") or {}
    reranker_enabled = bool(attack_changes.get("reranker"))
    if not reranker_enabled and any(r.get(RERANKER_KEY) is not None for r in results):
        reranker_enabled = True

    stage_plan = build_stage_plan(reranker_enabled)
    stage_sources = dict(stage_plan)

    rows: List[Dict[str, Any]] = []
    for result in results:
        qid = result.get("id", "")
        passages = get_query_adv_passages(qid, adv_docs)
        if passages is None:
            continue

        row = build_query_summary(passages, golden_by_query.get(str(qid), set()), result, corpus, stage_plan)
        row["query_id"] = qid
        rows.append(row)

    # ---- per query -------------------------------------------------------
    query_level = []
    for row in rows:
        query_level.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "chunk_retention": {
                stage: stage_block(row["totals"], row["retained"][stage], stage_sources[stage])
                for stage in stage_sources
            },
        })

    # ---- pooled across all scored queries ---------------------------------
    pooled_totals = {cat: sum(r["totals"][cat] for r in rows) for cat in CATEGORIES}
    pooled_retained = {
        stage: {cat: sum(r["retained"][stage][cat] for r in rows) for cat in CATEGORIES}
        for stage in stage_sources
    }

    # Overall section keeps only the retention rates (counts stay in the per-query blocks).
    overall_chunk_retention = {
        stage: {
            key: value
            for key, value in stage_block(pooled_totals, pooled_retained[stage], stage_sources[stage]).items()
            if key.endswith("_retention_rate")
        }
        for stage in stage_sources
    }

    return {
        "meta": {
            "pipeline": "chunker_with_reranker" if reranker_enabled else "chunker_without_reranker",
            "reranker_enabled": reranker_enabled,
            "stage_sources": stage_sources,
        },
        "overall": {
            "chunk_retention": overall_chunk_retention,
        },
        "queries": query_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute adversarial and golden chunk-retention metrics from ad corpus and pipeline output."
    )
    parser.add_argument("--ad-file", "-ad-file", dest="ad_file", required=True,
                        help="*_ad.json file with attribute_document and bridge_document")
    parser.add_argument("--corpus-file", "-corpus-file", dest="corpus_file", required=True,
                        help="chunked corpus JSON list with title/chunk/text records")
    parser.add_argument("--output-file", "-output-file", dest="output_file", required=True,
                        help="pipeline output JSON containing results")
    parser.add_argument("--save-to", dest="save_to", default=None,
                        help="Optional path to write the computed metrics JSON to disk")
    args = parser.parse_args()

    metrics = compute_metrics(args.ad_file, args.corpus_file, args.output_file)
    json_text = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.save_to:
        save_path = Path(args.save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json_text + "\n", encoding="utf-8")
    print(json_text)


if __name__ == "__main__":
    main()