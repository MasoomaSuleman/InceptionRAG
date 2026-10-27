#!/usr/bin/env python3
"""
Build a custom Wikipedia corpus from the query->corpus mapping in HotpotQA qrels.

This script uses the actual dataset qrels files to map each query-id to a set of
corpus-id values, then fetches each Wikipedia page by the canonical page ID:
  https://en.wikipedia.org/wiki?curid=[corpus-id]

Example:
    python data/build_query_wiki_corpus.py \
      --queries /tmp/masoo20/inception_rag/samples/hotpotqa_queries_100.json \
      --qrels-dir /tmp/masoo20/inception_rag/datasets/hotpotqa/qrels \
      --output-dir /tmp/masoo20/inception_rag/custom_corpus \
      --dataset hotpotqa

This script produces:
  - {dataset}_wiki_corpus_texts.json
  - {dataset}_wiki_contriever_embs.npy

which can be fed directly into run_attack.py via --corpus-texts and --corpus-embs.
"""

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import requests
import torch
from tqdm import tqdm

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "InceptionRAG/0.1 (debugging; contact: local)"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split long text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + chunk_size - overlap, end - overlap)
    return chunks


def load_queries(path: Path) -> List[Dict[str, str]]:
    """Load query records from a JSON file.

    Expected entries may have fields like: {"id": "abc", "question": "..."}
    or {"query_id": "abc", "question": "..."}.
    """
    with open(path, "r") as f:
        data = json.load(f)

    records: List[Dict[str, str]] = []
    for item in data:
        qid = item.get("id") or item.get("query_id") or item.get("_id")
        question = item.get("question") or item.get("query") or item.get("text")
        if qid is not None and question is not None:
            records.append({
                "id": str(qid).strip(),
                "question": str(question).strip(),
            })
    return records


def load_qrels_by_query(qrels_paths: Iterable[Path]) -> Dict[str, List[str]]:
    """Build query_id -> list of corpus_ids using train/dev/test qrels files.

    BEIR qrels are often plain TSV files with no header and the layout:
        query-id    corpus-id    score
    but some files may include a header row. This parser supports both.
    """
    corpus_ids_by_query: Dict[str, Set[str]] = defaultdict(set)

    for qrels_path in qrels_paths:
        qrels_path = Path(qrels_path)
        if not qrels_path.exists():
            continue

        with open(qrels_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            rows = [row for row in reader if row and any(cell.strip() for cell in row)]

        if not rows:
            continue

        # Skip a header row when present.
        header = [cell.strip().lower() for cell in rows[0]]
        if any(h in {"query-id", "query_id", "queryid", "qid", "query id"} for h in header):
            rows = rows[1:]

        for row in rows:
            if len(row) < 2:
                continue

            qid = str(row[0]).strip()
            corpus_id = str(row[1]).strip()
            if not qid or not corpus_id:
                continue
            corpus_ids_by_query[qid].add(corpus_id)

    return {qid: sorted(corpus_ids) for qid, corpus_ids in corpus_ids_by_query.items()}


def annotate_queries_with_corpus_ids(
    queries: List[Dict[str, str]],
    qrels_dir: Path,
) -> List[Dict[str, str]]:
    """Attach qrels-derived corpus_ids to each query record for debugging."""
    qrels_paths = [
        qrels_dir / "train.tsv",
        qrels_dir / "dev.tsv",
        qrels_dir / "test.tsv",
    ]
    qrels_mapping = load_qrels_by_query(qrels_paths)

    annotated: List[Dict[str, str]] = []
    for record in queries:
        updated = dict(record)
        qid = str(updated.get("id") or updated.get("query_id") or "").strip()
        updated["corpus_ids"] = qrels_mapping.get(qid, [])
        annotated.append(updated)

    return annotated


def write_queries_debug_file(path: Path, queries: List[Dict[str, str]]) -> None:
    """Persist the annotated query list back to the sample JSON file."""
    with open(path, "w") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


def download_article(
    session: requests.Session,
    corpus_id: str,
    output_dir: Path,
    retries: int = 3,
    overwrite: bool = False,
) -> str:
    """Download one Wikipedia plaintext extract and save it to disk."""
    corpus_id = str(corpus_id).strip()
    if not corpus_id.isdigit():
        raise ValueError(f"Corpus ID is not numeric: {corpus_id!r}")

    article_dir = output_dir / corpus_id
    text_path = article_dir / "text.txt"
    if text_path.exists() and not overwrite:
        text = text_path.read_text(encoding="utf-8").strip()
        if text:
            return text

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
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
                headers={"User-Agent": WIKIPEDIA_USER_AGENT},
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = 5.0
                if retry_after is not None:
                    try:
                        wait_seconds = max(float(retry_after), 5.0)
                    except ValueError:
                        pass
                if attempt < retries:
                    time.sleep(wait_seconds + random.uniform(0, 2))
                    continue
                raise RuntimeError(f"Wikipedia rate-limited corpus ID {corpus_id} after retries")

            response.raise_for_status()
            payload = response.json()
            pages = payload.get("query", {}).get("pages", [])
            if isinstance(pages, dict):
                pages = list(pages.values())
            if not pages:
                raise ValueError(f"No page payload returned for corpus ID {corpus_id}")

            page = pages[0]
            if page.get("missing") is True:
                raise ValueError(f"Wikipedia page not found for corpus ID {corpus_id}")

            text = page.get("extract", "").strip()
            if not text:
                raise ValueError(f"Wikipedia page has no text for corpus ID {corpus_id}")

            article_dir.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text + "\n", encoding="utf-8")
            return text
        except Exception as exc:  # pragma: no cover - network retries are exercised in practice
            last_error = exc
            if attempt < retries:
                backoff = 2 ** attempt + random.uniform(0, 1)
                if isinstance(exc, RuntimeError) and "rate-limited" in str(exc).lower():
                    backoff = max(backoff, 5.0)
                time.sleep(backoff)
                continue
            break

    print(f"[warn] skipping corpus ID {corpus_id}; failed to fetch Wikipedia text: {last_error}")
    return ""


def fetch_wikipedia_page_text(curid: str, output_dir: Path | None = None, retries: int = 3) -> str:
    """Fetch the plain-text extract for a page using the Wikipedia page ID.

    We use the canonical page identifier from qrels and resolve it through the
    MediaWiki API rather than searching by title text.
    """
    session = requests.Session()
    if output_dir is not None:
        return download_article(session, curid, output_dir, retries=retries, overwrite=False)

    try:
        response = session.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "pageids": str(curid),
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
            },
            timeout=30,
            headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        for page in pages:
            if page.get("missing") is True:
                continue
            text = page.get("extract", "").strip()
            if text:
                return text
    except Exception:
        pass

    return ""


def build_query_wiki_corpus(
    queries: List[Dict[str, str]],
    qrels_dir: Path,
    output_dir: Path,
    dataset_name: str,
    chunk_size: int = 500,
) -> Tuple[List[str], np.ndarray]:
    """Build a Wikipedia corpus from all qrels corpus ids attached to each query id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wiki_cache_dir = output_dir / "wiki_pages"
    wiki_cache_dir.mkdir(parents=True, exist_ok=True)

    annotated_queries = annotate_queries_with_corpus_ids(queries, qrels_dir)
    query_ids = {record["id"] for record in annotated_queries if record.get("id")}
    required_queries = {record["id"]: record.get("corpus_ids", []) for record in annotated_queries if record.get("id")}

    texts: List[str] = []
    seen_corpus_ids: Set[str] = set()

    print(f"[1/3] Mapping {len(query_ids)} queries to Wikipedia page ids...")
    for record in tqdm(annotated_queries, desc="Queries"):
        qid = record.get("id")
        if not qid:
            continue
        for corpus_id in required_queries.get(qid, []):
            if corpus_id in seen_corpus_ids:
                continue
            seen_corpus_ids.add(corpus_id)
            article_dir = wiki_cache_dir / corpus_id
            article_path = article_dir / "text.txt"
            if article_path.exists():
                cached_text = article_path.read_text(encoding="utf-8").strip()
                if cached_text:
                    print(f"[debug] using cached article for corpus_id={corpus_id} at {article_path}")
                    for chunk in chunk_text(cached_text, chunk_size=chunk_size, overlap=80):
                        if chunk.strip():
                            texts.append(chunk)
                    continue
            page_text = fetch_wikipedia_page_text(corpus_id, output_dir=wiki_cache_dir)
            if not page_text:
                print(f"[debug] missing text for corpus_id={corpus_id} saved_to={article_path}")
                continue
            print(f"[debug] saved article for corpus_id={corpus_id} at {article_path}")
            for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=80):
                if chunk.strip():
                    texts.append(chunk)

    if not texts:
        raise RuntimeError(
            "No Wikipedia text was retrieved. Check that the qrels files contain matching query-id -> corpus-id entries."
        )

    print(f"[2/3] Encoding {len(texts)} chunks with Contriever...")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []
    for i in tqdm(range(0, len(texts), 32), desc="Encoding"):
        batch = texts[i:i + 32]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outs = model(**inputs)
            embs = outs.last_hidden_state.mean(dim=1).cpu().numpy()

        embeddings.append(embs)

    embs = np.vstack(embeddings)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    text_path = output_dir / f"{dataset_name}_wiki_corpus_texts.json"
    emb_path = output_dir / f"{dataset_name}_wiki_contriever_embs.npy"

    with open(text_path, "w") as f:
        json.dump(texts, f, ensure_ascii=False)
    np.save(emb_path, embs)

    print(f"[3/3] Saved corpus to {text_path}")
    print(f"       Saved embeddings to {emb_path}")
    print(f"       Total chunks: {len(texts)}")
    return texts, embs


def main():
    parser = argparse.ArgumentParser(
        description="Build a Wikipedia corpus for the query set using qrels query-id -> corpus-id mappings"
    )
    parser.add_argument("--queries", type=str, required=True, help="JSON file with query objects containing an id field")
    parser.add_argument("--qrels-dir", type=str, required=True, help="Directory containing train.tsv/dev.tsv/test.tsv")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write corpus texts and embeddings")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset name used for output files")
    parser.add_argument("--chunk-size", type=int, default=500, help="Max characters per chunk")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    queries = load_queries(queries_path)
    annotated_queries = annotate_queries_with_corpus_ids(queries, Path(args.qrels_dir))
    write_queries_debug_file(queries_path, annotated_queries)

    build_query_wiki_corpus(
        queries=annotated_queries,
        qrels_dir=Path(args.qrels_dir),
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
