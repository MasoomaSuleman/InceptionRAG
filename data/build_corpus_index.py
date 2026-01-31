#!/usr/bin/env python3
"""
Build corpus embedding index for InceptionRAG retrieval.

This script creates the corpus embeddings and text cache required by the attack scripts.

Usage:
    python build_corpus_index.py \
        --dataset nq \
        --corpus-path /path/to/corpus.jsonl \
        --output-dir /path/to/cache \
        --retriever contriever
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def build_index(
    corpus_path: Path,
    output_dir: Path,
    dataset_name: str,
    retriever: str = "contriever",
    batch_size: int = 64,
    limit: int = None,
):
    """Build corpus embedding index."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load retriever
    print(f"\n[1/3] Loading {retriever} model...")
    if retriever == "contriever":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever")
    elif retriever == "contriever-ms":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        model = AutoModel.from_pretrained("facebook/contriever-msmarco")
    elif retriever == "ance":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-roberta-base-ance-firstp")
        model = AutoModel.from_pretrained("sentence-transformers/msmarco-roberta-base-ance-firstp")
    else:
        raise ValueError(f"Unknown retriever: {retriever}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  Device: {device}")
    
    # Load corpus
    print(f"\n[2/3] Loading corpus from {corpus_path}...")
    texts = []
    with open(corpus_path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            # Support different corpus formats
            text = item.get("text") or item.get("contents") or item.get("passage") or ""
            texts.append(text)
    
    print(f"  Loaded {len(texts)} documents")
    
    # Encode corpus
    print(f"\n[3/3] Encoding corpus (batch_size={batch_size})...")
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Normalize
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        embeddings.append(embs)
    
    embeddings = np.vstack(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Save
    suffix = "full" if limit is None else str(limit)
    emb_path = output_dir / f"{dataset_name}_{retriever}_{suffix}_embs.npy"
    text_path = output_dir / f"{dataset_name}_corpus_{suffix}_texts.json"
    
    print(f"\nSaving embeddings to {emb_path}...")
    np.save(emb_path, embeddings)
    
    print(f"Saving texts to {text_path}...")
    with open(text_path, "w") as f:
        json.dump(texts, f)
    
    print(f"\n✓ Done!")
    print(f"  Embeddings: {emb_path}")
    print(f"  Texts: {text_path}")


def main():
    parser = argparse.ArgumentParser(description="Build corpus embedding index")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (nq, hotpotqa, msmarco)")
    parser.add_argument("--corpus-path", type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--retriever", type=str, default="contriever", 
                        choices=["contriever", "contriever-ms", "ance"],
                        help="Retriever model")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    
    args = parser.parse_args()
    
    build_index(
        corpus_path=Path(args.corpus_path),
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset.lower(),
        retriever=args.retriever,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
