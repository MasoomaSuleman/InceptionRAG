"""
InceptionRAG - Dataset Download and Preparation

Downloads BEIR datasets (NQ, HotpotQA, MS MARCO) and prepares corpus embeddings.

Usage:
    python download_datasets.py --dataset nq
    python download_datasets.py --dataset hotpotqa
    python download_datasets.py --dataset msmarco
    python download_datasets.py --all
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# BEIR download URL
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

# Supported datasets
DATASETS = ["nq", "hotpotqa", "msmarco"]


def download_beir_dataset(dataset: str, out_dir: Path) -> Path:
    """
    Download and extract BEIR dataset.
    
    Args:
        dataset: Dataset name (nq, hotpotqa, msmarco)
        out_dir: Output directory
        
    Returns:
        Path to extracted dataset
    """
    from beir import util
    
    url = BEIR_URL.format(dataset)
    data_path = out_dir / dataset
    
    if not data_path.exists():
        logging.info(f"Downloading {dataset} from {url}...")
        data_path = Path(util.download_and_unzip(url, str(out_dir)))
        logging.info(f"Downloaded to {data_path}")
    else:
        logging.info(f"Dataset {dataset} already exists at {data_path}")
    
    return data_path


def load_beir_dataset(data_path: Path, split: str = "test") -> Tuple[Dict, Dict, Dict]:
    """
    Load BEIR dataset using GenericDataLoader.
    
    Returns:
        (corpus, queries, qrels)
    """
    from beir.datasets.data_loader import GenericDataLoader
    
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split=split)
    return corpus, queries, qrels


def encode_corpus_contriever(
    corpus: Dict,
    output_dir: Path,
    dataset_name: str,
    batch_size: int = 64,
    max_docs: int = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode corpus with Contriever and save embeddings.
    
    Args:
        corpus: BEIR corpus dict
        output_dir: Directory to save embeddings
        dataset_name: Name for cache files
        batch_size: Encoding batch size
        max_docs: Maximum documents to encode (None = all)
        
    Returns:
        (embeddings, texts)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{dataset_name}_contriever_embs.npy"
    text_path = output_dir / f"{dataset_name}_corpus_texts.json"
    
    # Check cache
    if emb_path.exists() and text_path.exists():
        logging.info(f"Loading cached embeddings from {emb_path}")
        embs = np.load(emb_path)
        with open(text_path, "r") as f:
            texts = json.load(f)
        return embs, texts
    
    # Load Contriever
    logging.info("Loading Contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logging.info(f"Contriever loaded on {device}")
    
    # Prepare texts
    corpus_ids = list(corpus.keys())
    if max_docs:
        corpus_ids = corpus_ids[:max_docs]
    
    texts = []
    for cid in corpus_ids:
        doc = corpus[cid]
        text = doc.get("title", "") + " " + doc.get("text", "")
        texts.append(text.strip())
    
    logging.info(f"Encoding {len(texts)} documents...")
    
    # Encode in batches - NO PADDING! Each doc encoded independently
    # batch_size=1 ensures no padding is needed
    all_embs = []
    for i in range(0, len(texts), 1):  # batch_size=1, NO PADDING
        doc = texts[i]
        inputs = tokenizer(
            [doc], 
            padding=False,  # NO PADDING!
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        all_embs.append(embs)
        
        if (i + 1) % 10000 == 0:
            logging.info(f"  Encoded {i + 1}/{len(texts)} documents")
    
    embs = np.vstack(all_embs)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    
    # Save
    np.save(emb_path, embs)
    with open(text_path, "w") as f:
        json.dump(texts, f)
    
    logging.info(f"Saved embeddings to {emb_path}")
    logging.info(f"Saved texts to {text_path}")
    
    return embs, texts


def prepare_dataset(
    dataset: str,
    base_dir: Path,
    max_corpus_docs: int = None,
    encode: bool = True,
):
    """
    Download and prepare a single dataset.
    
    Args:
        dataset: Dataset name
        base_dir: Base directory for data
        max_corpus_docs: Max corpus documents to encode
        encode: Whether to encode corpus embeddings
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Preparing dataset: {dataset}")
    logging.info(f"{'='*60}")
    
    # Download
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = download_beir_dataset(dataset, datasets_dir)
    
    # Load
    split = "train" if dataset == "msmarco" else "test"
    corpus, queries, qrels = load_beir_dataset(data_path, split)
    
    logging.info(f"Corpus: {len(corpus)} documents")
    logging.info(f"Queries: {len(queries)} queries")
    
    # Encode corpus
    if encode:
        corpus_dir = base_dir / "corpus"
        encode_corpus_contriever(corpus, corpus_dir, dataset, max_docs=max_corpus_docs)
    
    # Save queries sample
    samples_dir = base_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    query_list = []
    for qid, text in list(queries.items())[:100]:
        query_list.append({
            "id": qid,
            "question": text,
        })
    
    sample_path = samples_dir / f"{dataset}_queries_100.json"
    with open(sample_path, "w") as f:
        json.dump(query_list, f, indent=2)
    logging.info(f"Saved {len(query_list)} sample queries to {sample_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare BEIR datasets")
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--max-corpus", type=int, default=None, help="Max corpus docs to encode")
    parser.add_argument("--no-encode", action="store_true", help="Skip corpus encoding")
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    
    base_dir = Path(args.output_dir)
    
    if args.all:
        for ds in DATASETS:
            prepare_dataset(ds, base_dir, args.max_corpus, not args.no_encode)
    elif args.dataset:
        prepare_dataset(args.dataset, base_dir, args.max_corpus, not args.no_encode)
    else:
        parser.print_help()
        return
    
    logging.info("\n" + "="*60)
    logging.info("Dataset preparation complete!")
    logging.info("="*60)


if __name__ == "__main__":
    main()
