"""
InceptionRAG - Build Corpus Cache (Contriever Embeddings)

IMPORTANT: No padding! Each document is encoded independently.

Usage:
    python build_corpus_cache.py --dataset nq
    python build_corpus_cache.py --dataset hotpotqa
    python build_corpus_cache.py --dataset msmarco
    python build_corpus_cache.py --all
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def encode_corpus_contriever_no_padding(
    corpus_texts: List[str],
    output_dir: Path,
    dataset_name: str,
    batch_size: int = 1,  # Process one at a time to avoid padding
    device: str = "cuda",
) -> np.ndarray:
    """
    Encode corpus with Contriever WITHOUT padding.
    
    CRITICAL: Each document is encoded independently (batch_size=1) to avoid
    any padding that could affect embedding quality.
    
    Args:
        corpus_texts: List of corpus documents
        output_dir: Directory to save embeddings
        dataset_name: Name for cache files
        batch_size: Encoding batch size (default 1 = no padding)
        device: cuda or cpu
        
    Returns:
        embeddings array
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{dataset_name}_contriever_embs.npy"
    text_path = output_dir / f"{dataset_name}_corpus_texts.json"
    
    # Check cache
    if emb_path.exists() and text_path.exists():
        logging.info(f"Cache exists: {emb_path}")
        embs = np.load(emb_path)
        with open(text_path, "r") as f:
            texts = json.load(f)
        logging.info(f"Loaded {len(texts)} documents, embeddings shape: {embs.shape}")
        return embs
    
    logging.info(f"Building cache for {dataset_name}...")
    logging.info(f"Documents: {len(corpus_texts)}")
    logging.info(f"Device: {device}")
    logging.info(f"Batch size: {batch_size} (no padding)")
    
    # Load Contriever
    logging.info("Loading Contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").eval()
    model = model.to(device)
    
    # Encode documents one by one (NO PADDING!)
    all_embeddings = []
    
    logging.info("Encoding documents (no padding)...")
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding"):
        batch_texts = corpus_texts[i:i+batch_size]
        
        # Tokenize without padding (batch_size=1 means no padding needed)
        inputs = tokenizer(
            batch_texts,
            padding=False,  # NO PADDING!
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over tokens
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
        
        all_embeddings.append(embeddings)
        
        # Clear GPU memory periodically
        if i % 1000 == 0 and device == "cuda":
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Normalize
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / (norms + 1e-8)
    
    logging.info(f"Embeddings shape: {all_embeddings.shape}")
    
    # Save
    np.save(emb_path, all_embeddings)
    logging.info(f"Saved embeddings to {emb_path}")
    
    with open(text_path, "w") as f:
        json.dump(corpus_texts, f, ensure_ascii=False)
    logging.info(f"Saved texts to {text_path}")
    
    return all_embeddings


def load_beir_corpus(data_path: Path) -> List[str]:
    """Load corpus from BEIR dataset directory"""
    corpus_file = data_path / "corpus.jsonl"
    
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_file}")
    
    logging.info(f"Loading corpus from {corpus_file}...")
    
    texts = []
    with open(corpus_file, "r") as f:
        for line in f:
            doc = json.loads(line)
            # Combine title and text
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                combined = f"{title}\n{text}"
            else:
                combined = text
            texts.append(combined)
    
    logging.info(f"Loaded {len(texts)} documents")
    return texts


def build_cache(dataset: str, base_dir: Path, device: str = "cuda"):
    """Build cache for a single dataset"""
    logging.info("=" * 60)
    logging.info(f"Building cache for: {dataset}")
    logging.info("=" * 60)
    
    # Find dataset directory
    dataset_dir = base_dir / "datasets" / dataset
    if not dataset_dir.exists():
        logging.error(f"Dataset not found: {dataset_dir}")
        logging.error("Please run download first!")
        return False
    
    # Load corpus
    try:
        corpus_texts = load_beir_corpus(dataset_dir)
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return False
    
    # Build embeddings
    corpus_dir = base_dir / "corpus"
    encode_corpus_contriever_no_padding(
        corpus_texts,
        corpus_dir,
        dataset,
        batch_size=1,  # NO PADDING!
        device=device,
    )
    
    logging.info(f"Cache built successfully for {dataset}!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build Contriever corpus cache (no padding)")
    parser.add_argument("--dataset", type=str, choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--all", action="store_true", help="Build cache for all datasets")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    
    # Determine base directory
    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        base_dir = Path(__file__).parent.parent / "data"
    
    logging.info(f"Data directory: {base_dir}")
    
    datasets = ["nq", "hotpotqa", "msmarco"] if args.all else [args.dataset]
    
    if not datasets or datasets == [None]:
        parser.print_help()
        return
    
    for ds in datasets:
        success = build_cache(ds, base_dir, args.device)
        if not success:
            logging.error(f"Failed to build cache for {ds}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Cache building complete!")
    logging.info("=" * 60)
    
    # Show output files
    corpus_dir = base_dir / "corpus"
    if corpus_dir.exists():
        logging.info("\nOutput files:")
        for f in sorted(corpus_dir.glob("*")):
            size = f.stat().st_size / (1024 * 1024)  # MB
            logging.info(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
