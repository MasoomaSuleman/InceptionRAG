#!/usr/bin/env python3
"""Build the custom corpus bundle used by the attack pipeline.

Usage examples:
    # Uses the active config.yaml chunking settings automatically.
    python scripts/build_custom_chunked_corpus.py \
        --dataset hotpotqa \
        --ad-data /tmp/masoo20/inception_rag/adv/hotpotqa/hotpotqa_0-100_with_ad.json \
        --wiki-corpus /tmp/masoo20/inception_rag/custom_corpus/hotpotqa_wiki_corpus.txt \
        --output-dir /tmp/masoo20/inception_rag/custom_corpus
        --zoso-results /tmp/masoo20/inception_rag/suffix_pool/zoso_results/Qwen/Qwen2-5-3B-Instruct/hotpotqa_zoso_results.json

    # Override config.yaml values explicitly when needed.
    python scripts/build_custom_chunked_corpus.py \
        --dataset nq \
        --ad-data /tmp/masoo20/inception_rag/adv/nq/nq_0-100_with_ad.json \
        --wiki-corpus /tmp/masoo20/inception_rag/custom_corpus/nq_wiki_corpus.txt \
        --zoso-results /tmp/masoo20/inception_rag/suffix_pool/zoso_results/Qwen/Qwen2-5-3B-Instruct/nq_zoso_results.json \
        --chunking-strategy "token based" \
        --chunk-size 512 \
        --chunk-overlap 64

This script is designed for the "with-chunker" workflow that will be wired into
run_attack later. It performs the preparatory steps in one place:

1. Read the chunking configuration from config.yaml.
2. Load the wiki corpus text file from custom_corpus.
3. Build the adversarial corpus from the `_with_ad` dataset by applying ZOSO-style
   suffixes plus dual-ended injection to the attribute and bridge documents.
4. Merge wiki + adv texts.
5. Chunk according to the configured strategy (word-based or token-based).
6. Embed the chunked texts and store them as JSON + NumPy arrays under
   custom_corpus/chunked/.

Output file naming convention:
    custom_corpus/chunked/{dataset}-{strategy}-{chunk_size}-{chunk_overlap}.npy
    custom_corpus/chunked/{dataset}-{strategy}-{chunk_size}-{chunk_overlap}.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_CUSTOM_CORPUS_DIR = Path("/tmp/masoo20/inception_rag/custom_corpus")
DEFAULT_TMP_SUFFIX_POOL_DIR = Path("/tmp/masoo20/inception_rag/suffix_pool")
DEFAULT_DATASET = "hotpotqa"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "what", "when", "where", "who", "why", "how", "did", "does", "do",
    "it", "its", "this", "that", "these", "those", "be", "been", "being",
    "have", "has", "had", "will", "would", "could", "should", "can",
    "may", "might", "must", "shall", "about", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "which", "there", "here",
}



def load_yaml_config(path: str | Path | None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def normalize_chunking_strategy(value: str | None) -> str:
    strategy = (value or "word based").strip().lower().replace("-", " ")
    strategy = " ".join(strategy.split())
    if "token" in strategy:
        return "token"
    if "word" in strategy:
        return "word"
    raise ValueError(f"Unsupported chunking strategy: {value!r}. Use 'word based' or 'token based'.")


def resolve_chunking_config(config: dict, cli_strategy: str | None, cli_chunk_size: int | None, cli_overlap: int | None):
    raw_cfg = config.get("chunking") or config.get("corpus", {}).get("chunking") or {}
    strategy = cli_strategy or raw_cfg.get("strategy") or raw_cfg.get("chunking_strategy") or "word based"
    chunk_size = cli_chunk_size if cli_chunk_size is not None else raw_cfg.get("chunk_size") or raw_cfg.get("size") or 200
    chunk_overlap = cli_overlap if cli_overlap is not None else raw_cfg.get("chunk_overlap") or raw_cfg.get("overlap") or 20
    return normalize_chunking_strategy(strategy), int(chunk_size), int(chunk_overlap)


def _tokenize_text_for_word(text: str) -> List[str]:
    try:
        import nltk
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b\w+\b|[^\w\s]", text)


def _tokenize_text_for_token(text: str) -> List[str]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer.tokenize(text)
    except Exception:
        # Reasonable fallback when a tokenizer model is unavailable locally.
        return re.findall(r"\w+|[\.,!?;:()\[\]{}\"'/-]", text)


def _join_tokens(tokens: Sequence[str], strategy: str) -> str:
    if strategy == "word":
        return " ".join(tokens).strip()
    return "".join(tokens).strip()


def chunk_texts_by_strategy(texts: Sequence[str], strategy: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    tokenizer = _tokenize_text_for_word if strategy == "word" else _tokenize_text_for_token
    chunks: List[str] = []
    step = max(1, chunk_size - chunk_overlap)

    for text in texts:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        tokens = tokenizer(cleaned)
        if not tokens:
            continue

        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + chunk_size)
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue
            chunk_text = _join_tokens(chunk_tokens, strategy)
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(tokens):
                break

    return chunks


def extract_keywords(query: str) -> str:
    q = (query or "").lower()
    tokens = re.findall(r"\b\w+\b", q)
    keywords = [t for t in tokens if t not in STOPWORDS]
    return " ".join(keywords) if keywords else q


def inject_dual_ended(text: str, query: str, config: dict | None = None) -> str:
    cfg = config or {}
    if not cfg.get("dual_injection", True):
        return text

    query_prefix = cfg.get("query_prefix", True)
    separator = cfg.get("footer_separator", "\n")
    keywords = extract_keywords(query)
    footer = f"(Reference keywords: {keywords})"

    if not query_prefix:
        return f"{text}{separator}{footer}"
    return f"{query}\n\n{text}{separator}{footer}"


def inject_polymorphic_noise(suffix: str, enable: bool = True) -> str:
    if not enable or not suffix:
        return suffix
    random_id = f"x{np.random.randint(0, 0xFFFF):04X}"
    timestamp = f"T+{np.random.uniform(0, 1):.2f}s"
    return f"[{random_id} | {suffix} | {timestamp}]"


def load_ad_items(path: str | Path) -> List[dict]:
    data_path = Path(path)
    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("items") or payload.get("samples") or []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    raise ValueError(f"Unsupported AD JSON format in {data_path}")


def _sanitize_model_dir(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name or "default-model").strip("-_.")
    return sanitized or "default-model"


def _resolve_zoso_path(dataset: str, config: dict, zoso_path: str | None) -> Path | None:
    if zoso_path:
        zoso_file = Path(zoso_path)
        if zoso_file.exists():
            return zoso_file

    cfg_zoso = config.get("zoso") or config.get("dataset", {}).get("zoso") or {}
    candidate = cfg_zoso.get("path") or cfg_zoso.get("results_path")
    if candidate:
        z = Path(candidate)
        if z.exists():
            return z

    # Prefer the model-scoped ZOSO output, matching run_attack.py's discovery logic.
    llm_cfg = config.get("llm") or {}
    model_name = str(llm_cfg.get("model") or "").strip()
    model_dir = _sanitize_model_dir(model_name)
    if model_name:
        model_prefix = Path(model_name.split("/")[0]) if "/" in model_name else Path(model_dir)
        model_dir_candidates = [
            Path(model_dir),
            model_prefix / model_dir,
            Path(model_name.replace("/", "/")),
        ]
    else:
        model_dir_candidates = [Path("default-model")]

    candidates = [
        Path("/tmp/masoo20/inception_rag/suffix_pool/zoso_results") / model_dir / f"{dataset}_zoso_results.json",
        DEFAULT_TMP_SUFFIX_POOL_DIR / "zoso_results" / model_dir / f"{dataset}_zoso_results.json",
        PROJECT_ROOT / "data" / "suffix_pool" / "zoso_results" / model_dir / f"{dataset}_zoso_results.json",
        PROJECT_ROOT / "data" / "suffix_pool" / "zoso_results" / f"{dataset}_zoso_results.json",
        Path("/tmp/masoo20/inception_rag/data/suffix_pool/zoso_results") / f"{dataset}_zoso_results.json",
    ]

    # Flatten model-dir variations.
    for model_candidate in model_dir_candidates:
        candidates.extend([
            Path("/tmp/masoo20/inception_rag/suffix_pool/zoso_results") / model_candidate / f"{dataset}_zoso_results.json",
            DEFAULT_TMP_SUFFIX_POOL_DIR / "zoso_results" / model_candidate / f"{dataset}_zoso_results.json",
            PROJECT_ROOT / "data" / "suffix_pool" / "zoso_results" / model_candidate / f"{dataset}_zoso_results.json",
        ])

    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_zoso_suffixes(zoso_path: Path | None) -> tuple[str, str]:
    if zoso_path is None or not zoso_path.exists():
        return "", ""
    with open(zoso_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("best") or data.get("best_suffix") or data.get("best_pair") or {}
    anchor = str(best.get("anchor") or best.get("anchor_suffix") or "").strip()
    bridge = str(best.get("bridge") or best.get("bridge_suffix") or "").strip()
    return anchor, bridge


def build_adv_corpus_texts(
    items: Sequence[dict],
    dataset: str,
    zoso_path: str | Path | None = None,
    config: dict | None = None,
) -> List[str]:
    cfg = config or {}
    anchor_suffix, bridge_suffix = resolve_zoso_suffixes(_resolve_zoso_path(dataset, cfg, str(zoso_path) if zoso_path else None))
    texts: List[str] = []

    for item in items:
        ad = item.get("ad_text") or {}
        question = str(item.get("question") or item.get("text") or "").strip()
        text1 = str(ad.get("attribute_document") or ad.get("text1") or "").strip()
        text2 = str(ad.get("bridge_document") or ad.get("text2") or "").strip()

        if text1:
            improved = text1
            if anchor_suffix:
                improved = f"{improved} {inject_polymorphic_noise(anchor_suffix, cfg.get('noise_injection', True))}"
            improved = inject_dual_ended(improved, question, cfg)
            texts.append(improved)

        if text2:
            improved = text2
            if bridge_suffix:
                improved = f"{improved} {inject_polymorphic_noise(bridge_suffix, cfg.get('noise_injection', True))}"
            improved = inject_dual_ended(improved, question, cfg)
            texts.append(improved)

    return texts


def merge_wiki_and_adv_corpus(wiki_file: Path, adv_texts: Sequence[str]) -> List[str]:
    docs: List[str] = []
    if wiki_file.exists():
        wiki_text = wiki_file.read_text(encoding="utf-8")
        doc_candidates = re.split(r"\n\s*\n\s*---\s*\n\s*\n|\n\s*\n+", wiki_text)
        for doc in doc_candidates:
            if doc and doc.strip():
                docs.append(doc.strip())
    if not docs:
        wiki_text = wiki_file.read_text(encoding="utf-8") if wiki_file.exists() else ""
        if wiki_text.strip():
            docs = [wiki_text.strip()]

    merged = docs + [t.strip() for t in adv_texts if t and t.strip()]
    return [d for d in merged if d and d.strip()]


def embed_chunks(chunks: Sequence[str]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        embs = model.encode(list(chunks), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)
    except Exception:
        rng = np.random.default_rng(0)
        return np.asarray([rng.normal(0, 1, 768).astype(np.float32) for _ in chunks], dtype=np.float32)


def write_outputs(output_dir: Path, dataset: str, strategy: str, chunk_size: int, chunk_overlap: int, chunks: Sequence[str], embeddings: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}-{strategy}-{chunk_size}-{chunk_overlap}"
    json_path = output_dir / f"{stem}.json"
    npy_path = output_dir / f"{stem}.npy"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(chunks), f, ensure_ascii=False, indent=2)
    np.save(npy_path, embeddings)

    return json_path, npy_path


def main():
    parser = argparse.ArgumentParser(description="Build the custom corpus bundle for the chunker-aware attack pipeline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name used to resolve the AD file and output names")
    parser.add_argument("--ad-data", type=str, default=None, help="Path to the dataset JSON with ad_text entries, e.g. hotpotqa_100_with_ad.json")
    parser.add_argument("--wiki-corpus", type=str, default=None, help="Path to the wiki corpus .txt file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_CUSTOM_CORPUS_DIR), help="Directory for custom_corpus outputs")
    parser.add_argument("--chunking-strategy", type=str, default=None, help="Either 'word based' or 'token based'")
    parser.add_argument("--chunk-size", type=int, default=None, help="Length of each chunk in words/tokens")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap between adjacent chunks")
    parser.add_argument("--zoso-results", type=str, default=None, help="Optional override path to ZOSO JSON")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    strategy, chunk_size, chunk_overlap = resolve_chunking_config(config, args.chunking_strategy, args.chunk_size, args.chunk_overlap)

    project_root = PROJECT_ROOT
    dataset = args.dataset.lower()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Default data paths mirror the project layout.
    ad_path = Path(args.ad_data) if args.ad_data else project_root / "data" / "samples" / f"{dataset}_100_with_ad.json"
    wiki_path = Path(args.wiki_corpus) if args.wiki_corpus else output_dir / f"{dataset}_wiki_corpus.txt"

    if not ad_path.exists():
        raise FileNotFoundError(f"AD data not found: {ad_path}")
    if not wiki_path.exists():
        raise FileNotFoundError(f"Wiki corpus not found: {wiki_path}. Build it first via the custom corpus pipeline.")

    resolved_zoso = _resolve_zoso_path(dataset, config, args.zoso_results)
    if resolved_zoso is not None:
        print(f"Using ZOSO results: {resolved_zoso}")
    else:
        print(f"No ZOSO results found for dataset={dataset}; using empty suffixes.")

    raw_items = load_ad_items(ad_path)
    adv_texts = build_adv_corpus_texts(raw_items, dataset, resolved_zoso, config)
    adv_file = output_dir / f"{dataset}_adv_corpus.txt"
    adv_file.write_text("\n\n".join(t.strip() for t in adv_texts if t and t.strip()) + "\n", encoding="utf-8")
    print(f"Saved adversarial corpus txt to {adv_file}")

    merged_docs = merge_wiki_and_adv_corpus(wiki_path, adv_texts)
    final_chunks = chunk_texts_by_strategy(merged_docs, strategy, chunk_size, chunk_overlap)
    embeddings = embed_chunks(final_chunks)

    chunk_dir = output_dir / "chunked"
    json_path, npy_path = write_outputs(chunk_dir, dataset, strategy, chunk_size, chunk_overlap, final_chunks, embeddings)

    print(f"Chunking strategy: {strategy}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Final chunks: {len(final_chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved NumPy: {npy_path}")


if __name__ == "__main__":
    main()
