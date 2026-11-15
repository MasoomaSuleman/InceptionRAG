#!/usr/bin/env python3
"""Compute additional retrieval metrics from the adversarial JSON, chunked corpus, and pipeline output.

The script reads three inputs:
  --ad-file: *_ad.json list of entries, each containing ad_text.attribute_document and ad_text.bridge_document
  --corpus-file: chunked corpus JSON list of {title, chunk, text} records
  --output-file: pipeline output JSON containing results with retrieval_top_k_context / reranker_top_k_context

The script reports:
  - ACRR overall, adversarial-only, and bridge-only
  - adversarial chunk retrieval rate
  - content retained after chunking and in the final generation context
  - chunk survival after reranking
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def normalize_tokens(text: Any) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text or ""))]


def as_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("text", "content", "chunk_text", "passage"):
            if key in entry and entry[key] is not None:
                return str(entry[key])
        if "title" in entry and "text" in entry:
            return str(entry["text"]) or ""
        return ""
    return str(entry or "")


def get_query_ad_passages(item: Dict[str, Any], idx: int) -> List[Tuple[str, str, str]]:
    ad = item.get("ad_text") or {}
    docs = [
        ("attribute_document", ad.get("attribute_document") or ad.get("text1") or ""),
        ("bridge_document", ad.get("bridge_document") or ad.get("text2") or ""),
    ]
    passages: List[Tuple[str, str, str]] = []
    for label, text in docs:
        if not text:
            continue
        title = f"adv-{2 * idx + (0 if label == 'attribute_document' else 1)}"
        passages.append((label, title, str(text)))
    return passages


def count_selected_context_titles(entries: Sequence[Any]) -> Set[str]:
    titles: Set[str] = set()
    for entry in entries or []:
        if isinstance(entry, dict):
            title = entry.get("title") or entry.get("doc_id") or entry.get("id")
            if title is not None:
                titles.add(str(title))
        elif isinstance(entry, str):
            titles.add(entry)
    return titles


def chunk_pairs(entries: Sequence[Any]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title") or entry.get("doc_id") or entry.get("id")
        if title is None:
            continue
        chunk = entry.get("chunk")
        pairs.add((str(title), str(chunk) if chunk is not None else ""))
    return pairs


def get_selected_context(result: Dict[str, Any], reranker_on: bool) -> List[Any]:
    if reranker_on:
        for key in ("reranker_top_k_context", "final_context", "retrieval_top_k_context", "retrival_top_k_context"):
            value = result.get(key)
            if value is not None:
                return value if isinstance(value, list) else [value]
        return []

    for key in ("retrieval_top_k_context", "retrival_top_k_context"):
        value = result.get(key)
        if value is not None:
            return value if isinstance(value, list) else [value]
    return []


def get_first_k_context(result: Dict[str, Any]) -> List[Any]:
    for key in ("retrival_first_k_context", "retrieval_first_k_context"):
        value = result.get(key)
        if value is not None:
            return value if isinstance(value, list) else [value]
    return get_selected_context(result, reranker_on=False)


def compute_content_recall(source_text: str, chunk_texts: Iterable[str]) -> float:
    source_tokens = set(normalize_tokens(source_text))
    if not source_tokens:
        return 0.0
    chunk_tokens: Set[str] = set()
    for chunk in chunk_texts:
        chunk_tokens.update(normalize_tokens(chunk))
    if not chunk_tokens:
        return 0.0
    return len(source_tokens & chunk_tokens) / len(source_tokens)


def compute_content_recall_counts(source_text: str, chunk_texts: Iterable[str]) -> Dict[str, Any]:
    source_tokens = set(normalize_tokens(source_text))
    if not source_tokens:
        return {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0}
    chunk_tokens: Set[str] = set()
    for chunk in chunk_texts:
        chunk_tokens.update(normalize_tokens(chunk))
    matched = len(source_tokens & chunk_tokens)
    return {
        "matched_tokens": matched,
        "total_tokens": len(source_tokens),
        "ratio": matched / len(source_tokens) if source_tokens else 0.0,
    }


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_ad_records(path: str) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(f"Unsupported ad file format in {path}: expected a list of entries")


def load_corpus(path: str) -> Dict[str, List[Dict[str, Any]]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported corpus file format in {path}: expected a JSON list")

    by_title: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in payload:
        if not isinstance(record, dict):
            continue
        title = record.get("title")
        if title is None:
            continue
        by_title[str(title)].append(record)
    return by_title


def build_query_summary(
    ad_records: Sequence[Dict[str, Any]],
    corpus_by_title: Dict[str, List[Dict[str, Any]]],
    result: Dict[str, Any],
    query_index: int,
    reranker_on: bool,
) -> Dict[str, Any]:
    passages = get_query_ad_passages(ad_records[query_index], query_index)
    if not passages:
        return {
            "query_index": query_index,
            "question": result.get("question", ""),
            "attribute_passage_present": False,
            "bridge_passage_present": False,
            "attribute_chunk_rate": 0.0,
            "bridge_chunk_rate": 0.0,
            "attribute_content_retention_after_chunking": 0.0,
            "bridge_content_retention_after_chunking": 0.0,
            "attribute_content_retention_in_context": 0.0,
            "bridge_content_retention_in_context": 0.0,
            "query_adv_chunks": 0,
            "query_adv_chunks_found_in_selected_context": 0,
            "query_adv_chunks_found_in_first_k": 0,
            "query_adv_chunks_found_in_reranker": 0,
            "selected_context_titles": set(),
            "first_k_titles": set(),
            "reranker_titles": set(),
        }

    selected_context = get_selected_context(result, reranker_on)
    selected_titles = count_selected_context_titles(selected_context)

    first_k_context = get_first_k_context(result)
    first_k_titles = count_selected_context_titles(first_k_context)

    reranker_context = []
    if reranker_on:
        reranker_context = get_selected_context(result, reranker_on=True)
    reranker_titles = count_selected_context_titles(reranker_context)

    all_query_adv_titles: List[str] = []
    for _, title, _ in passages:
        all_query_adv_titles.append(title)

    query_adv_chunk_total = 0
    query_adv_found_selected = 0
    query_adv_found_first_k = 0
    query_adv_found_reranker = 0

    selected_pairs = chunk_pairs(selected_context)
    first_k_pairs = chunk_pairs(first_k_context)
    reranker_pairs = chunk_pairs(reranker_context)

    for title in all_query_adv_titles:
        chunk_records = corpus_by_title.get(title, [])
        corpus_pairs = {
            (str(rec.get("title") or ""), str(rec.get("chunk") if rec.get("chunk") is not None else ""))
            for rec in chunk_records
            if rec.get("title") is not None
        }
        if not corpus_pairs:
            corpus_pairs = {(title, "")}
        query_adv_chunk_total += len(corpus_pairs)

        hit_selected = corpus_pairs & selected_pairs
        hit_first_k = corpus_pairs & first_k_pairs
        hit_reranker = corpus_pairs & reranker_pairs

        query_adv_found_selected += len(hit_selected)
        query_adv_found_first_k += len(hit_first_k)
        query_adv_found_reranker += len(hit_reranker)

    selected_found_passage = 0
    selected_passage_details: List[Dict[str, Any]] = []
    for label, title, passage_text in passages:
        chunk_records = corpus_by_title.get(title, [])
        chunk_texts = [as_text(rec) for rec in chunk_records if as_text(rec)]
        if not chunk_texts:
            chunk_texts = [passage_text]

        title_pairs = {
            (str(rec.get("title") or ""), str(rec.get("chunk") if rec.get("chunk") is not None else ""))
            for rec in chunk_records
            if rec.get("title") is not None
        }
        if not title_pairs:
            title_pairs = {(title, "")}

        found = bool(title_pairs & selected_pairs)
        if found:
            selected_found_passage += 1

        selected_context_texts = [as_text(doc) for doc in selected_context if as_text(doc)]
        content_after_chunking_counts = compute_content_recall_counts(passage_text, chunk_texts)
        context_counts = compute_content_recall_counts(passage_text, selected_context_texts)
        selected_passage_details.append({
            "label": label,
            "title": title,
            "text": passage_text,
            "found_in_selected_context": found,
            "chunk_texts": chunk_texts,
            "content_retention_after_chunking": content_after_chunking_counts["ratio"],
            "content_retention_after_chunking_counts": content_after_chunking_counts,
            "content_retention_in_context": context_counts["ratio"],
            "content_retention_in_context_counts": context_counts,
        })

    query_summary = {
        "query_index": query_index,
        "question": result.get("question", ""),
        "attribute_passage_present": bool(selected_passage_details and selected_passage_details[0]["found_in_selected_context"]),
        "bridge_passage_present": bool(len(selected_passage_details) > 1 and selected_passage_details[1]["found_in_selected_context"]),
        "attribute_chunk_rate": 0.0,
        "bridge_chunk_rate": 0.0,
        "attribute_content_retention_after_chunking": 0.0,
        "bridge_content_retention_after_chunking": 0.0,
        "attribute_content_retention_after_chunking_counts": {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0},
        "bridge_content_retention_after_chunking_counts": {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0},
        "attribute_content_retention_in_context": 0.0,
        "bridge_content_retention_in_context": 0.0,
        "attribute_content_retention_in_context_counts": {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0},
        "bridge_content_retention_in_context_counts": {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0},
        "query_adv_chunks": query_adv_chunk_total,
        "query_adv_chunks_found_in_selected_context": query_adv_found_selected,
        "query_adv_chunks_found_in_first_k": query_adv_found_first_k,
        "query_adv_chunks_found_in_reranker": query_adv_found_reranker,
        "selected_context_titles": sorted(selected_titles),
        "first_k_titles": sorted(first_k_titles),
        "reranker_titles": sorted(reranker_titles),
    }

    for detail in selected_passage_details:
        label = detail["label"]
        if label == "attribute_document":
            query_summary["attribute_chunk_rate"] = 1.0 if detail["found_in_selected_context"] else 0.0
            query_summary["attribute_content_retention_after_chunking"] = detail["content_retention_after_chunking"]
            query_summary["attribute_content_retention_after_chunking_counts"] = detail["content_retention_after_chunking_counts"]
            query_summary["attribute_content_retention_in_context"] = detail["content_retention_in_context"]
            query_summary["attribute_content_retention_in_context_counts"] = detail["content_retention_in_context_counts"]
        elif label == "bridge_document":
            query_summary["bridge_chunk_rate"] = 1.0 if detail["found_in_selected_context"] else 0.0
            query_summary["bridge_content_retention_after_chunking"] = detail["content_retention_after_chunking"]
            query_summary["bridge_content_retention_after_chunking_counts"] = detail["content_retention_after_chunking_counts"]
            query_summary["bridge_content_retention_in_context"] = detail["content_retention_in_context"]
            query_summary["bridge_content_retention_in_context_counts"] = detail["content_retention_in_context_counts"]

    return query_summary


def compute_metrics(ad_path: str, corpus_path: str, output_path: str) -> Dict[str, Any]:
    ad_records = load_ad_records(ad_path)
    corpus_by_title = load_corpus(corpus_path)
    output = load_json(output_path)
    results = output.get("results", []) if isinstance(output, dict) else []

    reranker_enabled = bool(output.get("config_yaml", {}).get("attack_changes", {}).get("reranker"))
    if not reranker_enabled and any("reranker_top_k_context" in (result or {}) for result in results):
        reranker_enabled = True

    ttl_passages = 0
    found_passages = 0
    found_adv_passages = 0
    found_bridge_passages = 0
    total_adv_passages = 0
    total_bridge_passages = 0
    total_adv_chunks = 0
    retrieved_adv_chunks = 0
    total_reranker_adv_chunks = 0
    retrieved_reranker_adv_chunks = 0

    per_query_rows: List[Dict[str, Any]] = []

    for i, result in enumerate(results):
        query_summary = build_query_summary(ad_records, corpus_by_title, result, i, reranker_enabled)
        per_query_rows.append(query_summary)

        ttl_passages += 2
        total_adv_passages += 1
        total_bridge_passages += 1

        if query_summary["attribute_passage_present"]:
            found_adv_passages += 1
            found_passages += 1
        if query_summary["bridge_passage_present"]:
            found_bridge_passages += 1
            found_passages += 1

        total_adv_chunks += query_summary["query_adv_chunks"]
        retrieved_adv_chunks += query_summary["query_adv_chunks_found_in_first_k"]
        total_reranker_adv_chunks += query_summary["query_adv_chunks"]
        retrieved_reranker_adv_chunks += query_summary["query_adv_chunks_found_in_reranker"]

    total_queries = max(1, len(results))
    overall_acrr = found_passages / max(1, ttl_passages)
    adv_acrr = found_adv_passages / max(1, total_queries)
    bridge_acrr = found_bridge_passages / max(1, total_queries)

    chunk_rr = retrieved_adv_chunks / max(1, total_adv_chunks)
    rerank_survival = None
    if reranker_enabled:
        rerank_survival = retrieved_reranker_adv_chunks / max(1, total_reranker_adv_chunks)

    content_after_chunking = []
    content_in_context = []
    for row in per_query_rows:
        content_after_chunking.append((row["attribute_content_retention_after_chunking"] + row["bridge_content_retention_after_chunking"]) / 2)
        content_in_context.append((row["attribute_content_retention_in_context"] + row["bridge_content_retention_in_context"]) / 2)

    query_level = []
    for row in per_query_rows:
        query_entry = {
            "query_index": row["query_index"],
            "question": row["question"],
            "acrr": {
                "attribute_passage_present": bool(row["attribute_passage_present"]),
                "bridge_passage_present": bool(row["bridge_passage_present"]),
                "overall": (
                    1.0 if row["attribute_passage_present"] and row["bridge_passage_present"]
                    else 0.5 if row["attribute_passage_present"] or row["bridge_passage_present"]
                    else 0.0
                ),
                "counts": {
                    "found_passages": int(row["attribute_passage_present"]) + int(row["bridge_passage_present"]),
                    "total_passages": 2,
                },
            },
            "adversarial_chunk_retrieval_rate": {
                "value": (
                    (row["query_adv_chunks_found_in_first_k"] / max(1, row["query_adv_chunks"])) if row["query_adv_chunks"] else 0.0
                ),
                "counts": {
                    "retrieved": row["query_adv_chunks_found_in_first_k"],
                    "total": row["query_adv_chunks"],
                },
            },
            "content_retained_after_chunking": {
                "attribute": row["attribute_content_retention_after_chunking"],
                "bridge": row["bridge_content_retention_after_chunking"],
                "average": (
                    row["attribute_content_retention_after_chunking"] + row["bridge_content_retention_after_chunking"]
                ) / 2,
                "counts": {
                    "attribute": row.get("attribute_content_retention_after_chunking_counts", {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0}),
                    "bridge": row.get("bridge_content_retention_after_chunking_counts", {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0}),
                },
            },
            "content_retained_in_final_generation_context": {
                "attribute": row["attribute_content_retention_in_context"],
                "bridge": row["bridge_content_retention_in_context"],
                "average": (
                    row["attribute_content_retention_in_context"] + row["bridge_content_retention_in_context"]
                ) / 2,
                "counts": {
                    "attribute": row.get("attribute_content_retention_in_context_counts", {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0}),
                    "bridge": row.get("bridge_content_retention_in_context_counts", {"matched_tokens": 0, "total_tokens": 0, "ratio": 0.0}),
                },
            },
            "chunk_survival_after_reranking": {
                "value": (
                    (row["query_adv_chunks_found_in_reranker"] / max(1, row["query_adv_chunks"])) if reranker_enabled and row["query_adv_chunks"] else None
                ),
                "counts": {
                    "retrieved": row["query_adv_chunks_found_in_reranker"],
                    "total": row["query_adv_chunks"],
                },
            },
        }
        query_level.append(query_entry)

    summary = {
        "meta": {
            "ad_queries": len(ad_records),
            "result_queries": len(results),
            "reranker_enabled": reranker_enabled,
            "total_passages": ttl_passages,
        },
        "overall": {
            "acrr": {
                "overall": overall_acrr,
                "adversarial": adv_acrr,
                "bridge": bridge_acrr,
            },
            "adversarial_chunk_retrieval_rate": {
                "overall": chunk_rr,
            },
            "content_retained_after_chunking": {
                "overall": sum(content_after_chunking) / max(1, len(content_after_chunking)),
                "adversarial": sum(row["attribute_content_retention_after_chunking"] for row in per_query_rows) / max(1, len(per_query_rows)),
                "bridge": sum(row["bridge_content_retention_after_chunking"] for row in per_query_rows) / max(1, len(per_query_rows)),
            },
            "content_retained_in_final_generation_context": {
                "overall": sum(content_in_context) / max(1, len(content_in_context)),
                "adversarial": sum(row["attribute_content_retention_in_context"] for row in per_query_rows) / max(1, len(per_query_rows)),
                "bridge": sum(row["bridge_content_retention_in_context"] for row in per_query_rows) / max(1, len(per_query_rows)),
            },
            "chunk_survival_after_reranking": {
                "overall": rerank_survival,
            },
        },
        "queries": query_level,
        "summary": {
            "acrr": {
                "overall": overall_acrr,
                "adversarial": adv_acrr,
                "bridge": bridge_acrr,
            },
            "adversarial_chunk_retrieval_rate": {
                "overall": chunk_rr,
            },
            "content_retained_after_chunking": {
                "overall": sum(content_after_chunking) / max(1, len(content_after_chunking)),
                "adversarial": sum(row["attribute_content_retention_after_chunking"] for row in per_query_rows) / max(1, len(per_query_rows)),
                "bridge": sum(row["bridge_content_retention_after_chunking"] for row in per_query_rows) / max(1, len(per_query_rows)),
            },
            "content_retained_in_final_generation_context": {
                "overall": sum(content_in_context) / max(1, len(content_in_context)),
                "adversarial": sum(row["attribute_content_retention_in_context"] for row in per_query_rows) / max(1, len(per_query_rows)),
                "bridge": sum(row["bridge_content_retention_in_context"] for row in per_query_rows) / max(1, len(per_query_rows)),
            },
            "chunk_survival_after_reranking": {
                "overall": rerank_survival,
            },
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute adversarial retrieval and retention metrics from ad corpus and pipeline output.")
    parser.add_argument("--ad-file", "-ad-file", dest="ad_file", required=True, help="*_ad.json file with attribute_document and bridge_document")
    parser.add_argument("--corpus-file", "-corpus-file", dest="corpus_file", required=True, help="chunked corpus JSON list with title/chunk/text records")
    parser.add_argument("--output-file", "-output-file", dest="output_file", required=True, help="pipeline output JSON containing results")
    parser.add_argument("--save-to", dest="save_to", default=None, help="Optional path to write the computed metrics JSON to disk")
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
