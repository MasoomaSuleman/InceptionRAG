from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple, Optional


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def normalize_tokens(text: Any) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text or ""))]

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


# ============================================================================
# Retrieve data
# ============================================================================


def get_query_adv_passages(qid: str, adv_docs: Sequence[Dict]) -> Optional[List[Tuple[str, str, str]]]:
    query_adv_dict = next((d for d in adv_docs if d["id"] == qid), None)

    adv_text = query_adv_dict.get("ad_text", "") if query_adv_dict else None
    if adv_text:
        attribute_text = adv_text.get("attribute_document", "")
        bridge_text = adv_text.get("bridge_document", "")

        attribute_title = f"adv-{qid}-attribute"
        bridge_title = f"adv-{qid}-bridge"

        passages = [("attribute", attribute_title, attribute_text),
                    ("bridge", bridge_title, bridge_text)]
    else:
        passages = None

    return passages  


def get_context(result: Dict, keys: List) -> List:
    for key in keys:
        value = result.get(key)
        if value is not None:
            return value if isinstance(value, list) else [value]
    return []


def get_titles_chunk_pairs(entries: Sequence) -> Set[str]:
    title_chunk_paris: Set[Tuple[str, str]] = set()

    for entry in entries:
        if isinstance(entry, dict):
            title = entry.get("title") or entry.get("doc_id") or entry.get("id")
            if title is None:
                continue
            chunk = entry.get("chunk")
            title_chunk_paris.add(str(title), str(chunk) if chunk else "")

    return title_chunk_paris

def get_text(entry: Any) -> str:
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


# ============================================================================
# Counting
# ============================================================================

def content_recall_counts(source_text: str, chunk_texts: Iterable[str]) -> float:
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

def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


# ============================================================================
# Query Summaries
# ============================================================================

def build_query_summary(
        adv_passages: Tuple[str, str, str], 
        result: Dict,
        corpus: Dict,
        ):
    final_context = get_context(result, keys = ["reranker_top_k_context", "retrieval_top_k_context"])
    final_pairs = get_titles_chunk_pairs(final_context)
    final_texts = [get_text(doc) for doc in final_context if get_text(doc)]

    first_k_context = get_context(result, keys=["retrieval_first_k_context"])
    first_k_pairs = get_titles_chunk_pairs(first_k_context)

    query_adv_titles = [title for _, title, _ in adv_passages]

    query_adv_chunk_total = 0
    query_adv_found_final = 0
    query_adv_found_first_k = 0
    final_found_passage = 0

    final_passage_details: List[Dict] = []

    for label, title, text in adv_passages:
        chunk_records = corpus.get(title, [])
        chunk_texts = [get_text(rec) for rec in chunk_records if get_text(rec)]

        corpus_pairs = {
            (str(rec.get("title", "")), str(rec.get("chunk", "")))
            for rec in chunk_records
            if rec.get("title") is not None
        }

        if not corpus_pairs:
            corpus_pairs = {(title, "")}
        query_adv_chunk_total += len(corpus_pairs)

        final_hits = corpus_pairs & final_pairs
        first_k_hits = corpus_pairs & first_k_pairs

        query_adv_found_final += len(final_hits)
        query_adv_found_first_k += len(first_k_hits)
        final_found_passage += 1 if len(final_hits) else 0

        chunking_counts = content_recall_counts(text, chunk_texts)
        context_counts = content_recall_counts(text, final_texts)
        final_passage_details.append({
            "label": label,
            "title": title,
            "text": text,
            "found_in_final_context": bool(final_hits),
            "chunk_texts": chunk_texts,
            "content_retention_chunking": chunking_counts["ratio"],
            "content_retention_chunking_counts": chunking_counts,
            "content_retention_in_context": context_counts["ratio"],
            "content_retention_in_context_counts": context_counts,
        })

    query_summary = {
        "question": result.get("question", ""),
        "attribute_passage_present": bool(final_passage_details and final_passage_details[0]["found_in_selected_context"]),
        "bridge_passage_present": bool(len(final_passage_details) > 1 and final_passage_details[1]["found_in_selected_context"]),
        "query_adv_chunks": query_adv_chunk_total,
        "query_adv_chunks_found_in_final_context": query_adv_found_final,
        "query_adv_chunks_found_in_first_k": query_adv_found_first_k,
        "selected_context_titles": sorted([item[0] for item in final_pairs]),
        "first_k_titles": sorted([item[0] for item in first_k_pairs]),
    }

    for detail in final_passage_details:
        label = detail["label"]
        if label == "attribute_document":
            query_summary["attribute_chunk_rate"] = 1.0 if detail["found_in_selected_context"] else 0.0
            query_summary["attribute_content_retention_after_chunking"] = detail["content_retention_chunking"]
            query_summary["attribute_content_retention_after_chunking_counts"] = detail["content_retention_chunking_counts"]
            query_summary["attribute_content_retention_in_context"] = detacompute_content_recallil["content_retention_in_context"]
            query_summary["attribute_content_retention_in_context_counts"] = detail["content_retention_in_context_counts"]
        elif label == "bridge_document":
            query_summary["bridge_chunk_rate"] = 1.0 if detail["found_in_selected_context"] else 0.0
            query_summary["bridge_content_retention_after_chunking"] = detail["content_retention_chunking"]
            query_summary["bridge_content_retention_after_chunking_counts"] = detail["content_retention_chunking_counts"]
            query_summary["bridge_content_retention_in_context"] = detail["content_retention_in_context"]
            query_summary["bridge_content_retention_in_context_counts"] = detail["content_retention_in_context_counts"]

    return query_summary


def compute_metrics(adv_path: str, corpus_path: str, results_path: str):
    adv_docs = load_adv_docs(adv_path)
    corpus = load_corpus(corpus_path)
    output = load_json(results_path)
    results = output.get("results", []) if isinstance(output, dict) else []

    reranker_enabled = bool(output.get("config_yaml", {}).get("attack_changes", {}).get("reranker"))
    if not reranker_enabled and any("reranker_top_k_context" in (result or {}) for result in results):
        reranker_enabled = True

    per_query_rows: List[Dict] = []
    for result in results:
        qid = result.get("id", "")
        passages = get_query_adv_passages(qid, adv_docs)
        if passages is None:
            continue

        query_summary = build_query_summary(passages)
        query_summary["query_id"] = qid
        per_query_rows.append(query_summary)

    total_passages = 2 * len(results)
    total_queries = len(results)
    found_attr = sum(r["attribute_passage_present"] for r in per_query_rows)
    found_bridge = sum(r["bridge_passage_present"] for r in per_query_rows)

    total_adv_chunks = sum(r["query_adv_chunks"] for r in per_query_rows)
    retrieved_adv_chunks = sum(r["query_adv_chunks_found_in_first_k"] for r in per_query_rows)
    final_adv_chunks = sum(r["query_adv_chunks_found in final_context"] for r in per_query_rows)

    overall_acrr = (found_attr + found_bridge) / max(1, total_passages)
    attr_acrr = found_attr / total_queries
    bridge_acrr = found_bridge / total_queries
    chunk_rr = retrieved_adv_chunks / max(1, total_adv_chunks)
    rerank_survival = final_adv_chunks / max(1, total_adv_chunks) if reranker_enabled else None

    ctx_attr = [r["attribute_content_retention_in_context"] for r in per_query_rows]
    ctx_bridge = [r["bridge_content_retention_in_context"] for r in per_query_rows]
    ctx_overall = [(a + b) / 2 for a, b in zip(ctx_attr, ctx_bridge)]

    ctx_chunk_attr = [r["attribute_content_retention_after_chunking"] for r in per_query_rows]
    ctx_chunk_bridge = [r["bridge_content_retention_after_chunking"] for r in per_query_rows]
    ctx_chunk_overall = [(a + b) / 2 for a, b in zip(ctx_attr, ctx_bridge)]

    query_level = []
    for row in per_query_rows:
        present_a, present_b = row["attribute_passage_present"], row["bridge_passage_present"]
        total = row["query_adv_chunks"]
        query_level.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "acrr": {
                "attribute_passage_present": present_a,
                "bridge_passage_present": present_b,
                "overall": 1.0 if present_a and present_b else 0.5 if present_a or present_b else 0.0,
                "counts": {"found_passages": int(present_a) + int(present_b), "total_passages": 2},
            },
            "adversarial_chunk_retrieval_rate": {
                "value": row["query_adv_chunks_found_in_first_k"] / total if total else 0.0,
                "counts": {"retrieved": row["query_adv_chunks_found_in_first_k"], "total": total},
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
                    "attribute": row["attribute_content_retention_in_context_counts"],
                    "bridge": row["bridge_content_retention_in_context_counts"],
                },
            },
            "chunk_survival_after_reranking": {
                "value": row["query_adv_chunks_found_in_reranker"] / total if reranker_enabled and total else None,
                "counts": {"retrieved": row["query_adv_chunks_found_in_reranker"], "total": total},
            },
        })


    aggregate = {
        "acrr": {"overall": overall_acrr, "attribute": attr_acrr, "bridge": bridge_acrr},
        "adversarial_chunk_retrieval_rate": {"overall": chunk_rr},
        "content_retained_after_chunking": {
            "overall": mean(ctx_chunk_overall),
            "attribute": mean(ctx_chunk_attr),
            "bridge": mean(ctx_chunk_bridge),
        },
        "content_retained_in_final_generation_context": {
            "overall": mean(ctx_overall),
            "attribute": mean(ctx_attr),
            "bridge": mean(ctx_bridge),
        },
        "chunk_survival_after_reranking": {"overall": rerank_survival},
    }

    return {
        "meta": {
            "result_queries": len(results),
            "reranker_enabled": reranker_enabled,
            "total_passages": total_passages,
            "indexed_adversarial_chunks": total_adv_chunks,
        },
        "overall": aggregate,
        "queries": query_level,
        "summary": aggregate,
    }


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
