#!/usr/bin/env python3
"""
InceptionRAG - Attack Sample

使用预生成的 samples 和 suffix_pool 直接进行攻击测试
不需要运行完整 pipeline，直接测试攻击效果

Usage:
    python -m InceptionRAG.attack_sample --config config.yaml --dataset nq --limit 10
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModel

from .llm_client import LLMConfig, create_llm_client
from .run_attack import SuffixPool, TwoStageRetriever, InceptionRAGAttacker, InceptionRAGConfig


def main():
    parser = argparse.ArgumentParser(description="InceptionRAG Attack Sample")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, choices=["nq", "hotpotqa", "msmarco"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max samples to attack")
    parser.add_argument("--output", type=str, default=None, help="Output results JSON")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--retrieval-first-k", type=int, default=20)
    parser.add_argument("--cttft-neighbor-k", type=int, default=10)
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent
    
    # Load config
    llm_config = LLMConfig.from_yaml(args.config)
    config = InceptionRAGConfig(
        llm_config=llm_config,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_first_k=args.retrieval_first_k,
        cttft_neighbor_k=args.cttft_neighbor_k,
    )
    
    print("=" * 60)
    print("InceptionRAG Attack Sample")
    print("=" * 60)
    print(f"  Model: {llm_config.model}")
    print(f"  Dataset: {args.dataset}")
    
    ds = args.dataset.lower()
    model_folder = llm_config.model.replace(".", "-").replace(" ", "-")
    
    # Paths (使用 data/samples/ 和 data/suffix_pool/)
    sample_path = BASE_DIR / f"data/samples/{ds}_100_with_ad.json"
    suffix_pool_path = BASE_DIR / "data/suffix_pool/suffix_candidates.json"
    zoso_results_path = BASE_DIR / f"data/suffix_pool/zoso_results/{model_folder}/{ds}_zoso_results.json"
    corpus_embs_path = BASE_DIR / f"data/corpus/{ds}_contriever_embs.npy"
    corpus_texts_path = BASE_DIR / f"data/corpus/{ds}_corpus_texts.json"
    
    # Validate paths
    if not sample_path.exists():
        print(f"Error: Sample data not found at {sample_path}")
        return
    if not corpus_embs_path.exists():
        print(f"Error: Corpus embeddings not found at {corpus_embs_path}")
        print(f"Run: bash scripts/build_all_cache.sh")
        return
    
    # Load sample data
    print(f"\n[1/5] Loading samples from {sample_path}...")
    with open(sample_path, "r") as f:
        all_data = json.load(f)
    if args.limit:
        all_data = all_data[:args.limit]
    print(f"  Loaded: {len(all_data)} samples")
    
    # Load corpus
    print(f"\n[2/5] Loading corpus...")
    corpus_embs = np.load(corpus_embs_path)
    with open(corpus_texts_path, "r") as f:
        corpus_texts = json.load(f)
    print(f"  Corpus: {len(corpus_texts)} docs")
    
    # Load Contriever
    print(f"\n[3/5] Loading Contriever...")
    contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    contriever = AutoModel.from_pretrained("facebook/contriever").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    contriever = contriever.to(device)
    print(f"  Device: {device}")
    
    # Load suffix pool
    print(f"\n[4/5] Loading suffix pool...")
    suffix_pool = SuffixPool(suffix_pool_path)
    
    # Load ZOSO results
    optimal_anchor_idx = 0
    optimal_bridge_idx = 0
    if zoso_results_path.exists():
        with open(zoso_results_path, "r") as f:
            zoso = json.load(f)
        optimal_anchor_idx = zoso.get("best", {}).get("anchor_idx", 0)
        optimal_bridge_idx = zoso.get("best", {}).get("bridge_idx", 0)
        print(f"  ZOSO optimal: anchor={optimal_anchor_idx}, bridge={optimal_bridge_idx}")
    else:
        print(f"  Warning: ZOSO results not found, using default suffix (idx=0)")
    
    # Initialize attacker
    print(f"\n[5/5] Initializing attacker...")
    retriever = TwoStageRetriever(corpus_texts, corpus_embs, contriever, contriever_tokenizer, config)
    attacker = InceptionRAGAttacker(config, suffix_pool, retriever, optimal_anchor_idx, optimal_bridge_idx)
    
    # Attack
    print("\n" + "=" * 60)
    print("Running Attack")
    print("=" * 60)
    
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = attacker.attack_batch(all_data, output_path)
    
    # Final stats
    print("\n" + "=" * 60)
    print("Attack Complete!")
    print("=" * 60)
    print(f"Total: {attacker.stats['total']}")
    print(f"Success: {attacker.stats['attack_success']}")
    print(f"ASR: {attacker.stats['attack_success'] / attacker.stats['total'] * 100:.1f}%")
    print(f"L1 Success: {attacker.stats['l1_success']}")
    print(f"L2 Success: {attacker.stats['l2_success']}")
    print(f"L3 Success: {attacker.stats['l3_success']}")
    
    if output_path:
        print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
