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
import csv
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable

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
    

def load_qrels_by_corpus(qrels_paths: Iterable[Path]) -> Dict[str, List[str]]:
    """Build query_id -> list of corpus_ids using train/dev/test qrels files.

    BEIR qrels are often plain TSV files with no header and the layout:
        query-id    corpus-id    score
    but some files may include a header row. This parser supports both.
    """
    query_ids_by_corpus: Dict[str, Set[str]] = defaultdict(set)

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
            query_ids_by_corpus[corpus_id].add(qid)

    return {
        corpus_id: sorted(query_ids)
        for corpus_id, query_ids in query_ids_by_corpus.items()
    }


def load_queries(path: Path) -> List[Dict[str, str]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            items = (json.loads(line) for line in f if line.strip())
        else:
            items = json.load(f)

        for item in items:
            qid = item.get("id") or item.get("query_id") or item.get("_id")
            question = item.get("question") or item.get("query") or item.get("text")
            metadata = item.get("metadata", "")

            if qid is not None and question is not None:
                records.append({
                    "id": str(qid).strip(),
                    "question": str(question).strip(),
                    "metadata": metadata
                })

    return records


def encode_corpus_contriever(
    corpus: Dict,
    base_dir: Path,
    dataset_name: str,
    batch_size: int = 64,
    max_docs: int = None,
):
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
    
    output_dir = base_dir / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{dataset_name}_contriever_embs.npy"
    text_path = output_dir / f"{dataset_name}_corpus_texts.jsonl"
    
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
    
    qrels_dir = base_dir / f"datasets/{dataset_name}/qrels"
    qrels_paths = [
        qrels_dir / "train.tsv",
        qrels_dir / "dev.tsv",
        qrels_dir / "test.tsv",
    ]
    corpus_query_mapping = load_qrels_by_corpus(qrels_paths)
    texts = []
    
    for cid in corpus_ids:
        doc = corpus[cid]
        text = doc.get("title", "") + " " + doc.get("text", "")
        text_dict = {"title": doc.get("title", ""), "chunk": None, "text": text.strip()}
        if cid in corpus_query_mapping:
            query_ids = corpus_query_mapping[cid]
            text_dict["query_ids"] = query_ids
        texts.append(text_dict)
    
    if not emb_path.exists():
        logging.info(f"Encoding {len(texts)} documents...")
        
        # Encode in batches - NO PADDING! Each doc encoded independently
        # batch_size=1 ensures no padding is needed
        all_embs = []
        for i in range(0, len(texts), 1):  # batch_size=1, NO PADDING
            doc = texts[i].get("text", "")
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
    queries_path = base_dir / f"datasets/{dataset}/queries.jsonl"
    
    # Load
    split = "train" if dataset == "msmarco" else "test"
    corpus, queries, qrels = load_beir_dataset(data_path, split)
    
    qrels_dir = base_dir / f"datasets/{dataset}/qrels"
    qrels_paths = [
        qrels_dir / "train.tsv",
        qrels_dir / "dev.tsv",
        qrels_dir / "test.tsv",
    ]
    query_mapping = load_qrels_by_query(qrels_paths)
    queries = load_queries(queries_path)
    
    
    logging.info(f"Corpus: {len(corpus)} documents")
    logging.info(f"Queries: {len(queries)} queries")
    
    # Encode corpus
    if encode:
        
        encode_corpus_contriever(corpus, base_dir, dataset, max_docs=max_corpus_docs)
    
    # Save queries sample
    samples_dir = base_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    query_list = []
    for item in queries[:100]:
        qid = item.get("id", "")
        question = item.get("question", "")
        metadata = item.get("metadata", "")
        corpus_ids = list(query_mapping[qid])
        query_list.append({
            "id": qid,
            "question": question,
            "metadata": metadata,
            "corpus_ids": corpus_ids
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
