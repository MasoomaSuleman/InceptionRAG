#!/usr/bin/env python3
"""
Build a custom Wikipedia corpus for a query set and encode it with Contriever.

Example:
    python data/build_query_wiki_corpus.py \
      --queries /tmp/masoo20/inception_rag/samples/hotpotqa_queries_100.json \
      --output-dir /tmp/masoo20/inception_rag/custom_corpus \
      --dataset hotpotqa

This script:
  1. reads the query list
  2. searches Wikipedia for each query
  3. fetches the matched page text(s)
  4. chunks long pages into passages
  5. builds a Contriever corpus index (.npy + .json)

Then you can run the attack pipeline with:
  --corpus-embs /tmp/.../hotpotqa_query_wiki_contriever_embs.npy \
  --corpus-texts /tmp/.../hotpotqa_query_wiki_corpus_texts.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from urllib.parse import quote

import numpy as np
import requests
import torch
from tqdm import tqdm


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split long text into overlapping chunks of roughly chunk_size chars."""
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


def search_wikipedia_titles(query: str, limit: int = 3) -> List[str]:
    """Search Wikipedia for the best matching titles."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "utf8": 1,
        "srinfo": "totalhits",
    }
    try:
        rsp = requests.get(url, params=params, timeout=20)
        rsp.raise_for_status()
        data = rsp.json()
        results = data.get("query", {}).get("search", [])
        return [item.get("title", "") for item in results if item.get("title")]
    except Exception:
        return []


def fetch_wikipedia_page_text(title: str) -> str:
    """Fetch the plain-text extract for a Wikipedia page."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
        "exsectionformat": "plain",
    }
    try:
        rsp = requests.get(url, params=params, timeout=25)
        rsp.raise_for_status()
        data = rsp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return extract.strip()
    except Exception:
        pass
    return ""


def load_queries(path: Path) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)

    queries: List[str] = []
    for item in data:
        q = item.get("question") or item.get("query") or item.get("text")
        if q:
            queries.append(str(q).strip())
    return queries


def build_query_wiki_corpus(queries: List[str], output_dir: Path, dataset_name: str, chunk_size: int = 500) -> Tuple[List[str], np.ndarray]:
    """Build a corpus from Wikipedia pages relevant to the query list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    seen_titles: Set[str] = set()

    print(f"[1/3] Fetching Wikipedia pages for {len(queries)} queries...")
    for idx, q in enumerate(tqdm(queries), start=1):
        titles = search_wikipedia_titles(q, limit=3)
        for title in titles:
            title_key = title.lower().strip()
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            page_text = fetch_wikipedia_page_text(title)
            if not page_text:
                continue
            for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=80):
                if chunk.strip():
                    texts.append(chunk)

    if not texts:
        raise RuntimeError("No Wikipedia text was retrieved for the provided queries.")

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
    parser = argparse.ArgumentParser(description="Build a Wikipedia-based custom corpus for the query set")
    parser.add_argument("--queries", type=str, required=True, help="JSON file with query objects or strings")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write corpus texts and embeddings")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset name used for output files")
    parser.add_argument("--chunk-size", type=int, default=500, help="Max characters per chunk")
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    build_query_wiki_corpus(queries, Path(args.output_dir), args.dataset, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
