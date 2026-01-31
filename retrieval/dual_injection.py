"""
InceptionRAG - Stage 2: Retrieval Hijacking via Dual-Ended Injection

Dual-Ended Injection Strategy:
1. Subject Header: Prepend the query to poisoned documents to boost semantic similarity
2. Footer Metadata: Append stealthy keyword tags disguised as system metadata
3. Two-Stage Retrieval: First retrieve from corpus, then merge and re-rank with poisoned docs

Paper Reference:
    Stage 2: Retrieval Hijacking via Dual-Ended Injection
    - Subject Header augments the document's front with the target query
    - Footer Metadata appends stealthy keyword annotations
    - Two-stage retrieval ensures adversarial documents are prioritized
"""

import re
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Stopwords for Keyword Extraction
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "what", "when", "where", "who", "why", "how", "did", "does", "do",
    "it", "its", "this", "that", "these", "those", "be", "been", "being",
    "have", "has", "had", "will", "would", "could", "should", "can",
    "may", "might", "must", "shall", "about", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "which", "there", "here"
}


# ============================================================================
# Footer Metadata Templates (Stealthy Query Insertion)
# ============================================================================

RETRIEVAL_TEMPLATES = [
    "(Reference keywords: {keywords})",
    "Index terms: {keywords}",
    "Document classification terms: {keywords}",
    "Topic annotation: {keywords}",
    "Note: for internal tagging purposes, this text is categorized under: {keywords}.",
    "Metadata: {keywords}",
    "Additional relevance tags: {keywords}",
    "Related topics: {keywords}",
    "Content tags: {keywords}",
]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DualInjectionConfig:
    """Configuration for Dual-Ended Injection"""
    # Subject header (query prefix)
    enable_subject_header: bool = True
    
    # Footer metadata
    enable_footer_metadata: bool = True
    footer_separator: str = "\n\n---\n"
    use_stopwords_filter: bool = True
    
    # Retrieval parameters
    retrieval_top_k: int = 10           # Final top-k documents
    retrieval_first_k: int = 20         # First stage retrieval count


# ============================================================================
# Dual-Ended Injector
# ============================================================================

class DualEndedInjector:
    """
    Dual-Ended Injection for Retrieval Hijacking.
    
    Injects both subject header (query prefix) and footer metadata (keyword tags)
    into poisoned documents to maximize retrieval priority.
    
    Format: {query}\n\n{poisoned_content}\n\n---\n{metadata_template.format(keywords)}
    """
    
    def __init__(self, config: DualInjectionConfig = None):
        self.config = config or DualInjectionConfig()
    
    def extract_keywords(self, query: str) -> str:
        """
        Extract keywords from query by removing stopwords.
        Preserves word order (important for Contriever).
        
        Args:
            query: The target query
            
        Returns:
            Space-separated keywords string
        """
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        if self.config.use_stopwords_filter:
            keywords = [t for t in tokens if t not in STOPWORDS]
        else:
            keywords = tokens
        
        # Fallback: if all words filtered, return original query
        if not keywords:
            return query
        
        return " ".join(keywords)
    
    def inject(self, text: str, query: str) -> str:
        """
        Apply dual-ended injection to a poisoned document.
        
        Structure:
        - Subject Header: Full query (boosts semantic similarity)
        - Original Content: The poisoned text
        - Footer Metadata: Extracted keywords in metadata template
        
        Args:
            text: The poisoned document (Attribute/Bridge document with suffix)
            query: The target query
            
        Returns:
            Injected document with dual-ended augmentation
        """
        result = text
        
        # Subject Header: Prepend query
        if self.config.enable_subject_header:
            result = f"{query}\n\n{result}"
        
        # Footer Metadata: Append keyword tags
        if self.config.enable_footer_metadata:
            keywords = self.extract_keywords(query)
            template = random.choice(RETRIEVAL_TEMPLATES)
            footer = template.format(keywords=keywords)
            result = f"{result}{self.config.footer_separator}{footer}"
        
        return result
    
    def inject_batch(self, texts: List[str], query: str) -> List[str]:
        """Apply dual-ended injection to multiple documents"""
        return [self.inject(text, query) for text in texts]


# ============================================================================
# Two-Stage Retriever
# ============================================================================

class TwoStageRetriever:
    """
    Two-Stage Retrieval for InceptionRAG.
    
    Stage 1: Retrieve first_k documents from corpus
    Stage 2: Merge with poisoned documents, re-rank all, select top_k
    
    This ensures poisoned documents compete with retrieved documents
    and are prioritized when semantically similar to the query.
    """
    
    def __init__(
        self,
        corpus_texts: List[str],
        corpus_embs: np.ndarray,
        contriever,
        contriever_tokenizer,
        config: DualInjectionConfig = None,
    ):
        self.corpus_texts = corpus_texts
        self.corpus_embs = corpus_embs
        self.contriever = contriever
        self.contriever_tokenizer = contriever_tokenizer
        self.config = config or DualInjectionConfig()
        self.device = next(contriever.parameters()).device
        
        print(f"  Two-Stage Retriever initialized")
        print(f"  Corpus: {len(corpus_texts)} docs")
        print(f"  Retrieval: first-{self.config.retrieval_first_k} -> top-{self.config.retrieval_top_k}")
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text with Contriever (no padding for accuracy)"""
        inputs = self.contriever_tokenizer(
            [text], padding=False, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            emb = self.contriever(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    def retrieve(self, question: str, poisoned_texts: List[str]) -> Tuple[List[str], int]:
        """
        Two-stage retrieval with poisoned document injection.
        
        Args:
            question: The target query
            poisoned_texts: List of poisoned documents (with dual-ended injection applied)
            
        Returns:
            (retrieved_docs, adv_retrieved_count): Final documents and count of poisoned docs retrieved
        """
        # Encode query
        query_emb = self._encode(question)
        
        # Stage 1: Retrieve first_k from corpus
        corpus_scores = self.corpus_embs @ query_emb
        first_k = self.config.retrieval_first_k
        top_indices = np.argsort(corpus_scores)[::-1][:first_k]
        
        # Stage 2: Merge with poisoned docs, re-rank
        num_corpus = len(top_indices)
        candidate_texts = [self.corpus_texts[i] for i in top_indices] + poisoned_texts
        
        # Encode all candidates
        candidate_embs = []
        for text in candidate_texts:
            emb = self._encode(text)
            candidate_embs.append(emb)
        
        candidate_embs = np.array(candidate_embs)
        candidate_scores = candidate_embs @ query_emb
        
        # Select top_k
        top_k = self.config.retrieval_top_k
        final_indices = np.argsort(candidate_scores)[::-1][:top_k]
        
        # Count poisoned docs retrieved (indices >= num_corpus are poisoned)
        adv_retrieved_count = sum(1 for idx in final_indices if idx >= num_corpus)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        return [candidate_texts[i] for i in final_indices], adv_retrieved_count
    
    def retrieve_batch(
        self, 
        questions: List[str], 
        poisoned_texts_list: List[List[str]]
    ) -> List[Tuple[List[str], int]]:
        """Batch retrieval for multiple queries"""
        results = []
        for q, p in zip(questions, poisoned_texts_list):
            results.append(self.retrieve(q, p))
        return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Demo of dual-ended injection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual-Ended Injection Demo")
    parser.add_argument("--query", type=str, required=True, help="Target query")
    parser.add_argument("--text", type=str, required=True, help="Poisoned document text")
    args = parser.parse_args()
    
    config = DualInjectionConfig()
    injector = DualEndedInjector(config)
    
    result = injector.inject(args.text, args.query)
    
    print("=" * 60)
    print("Dual-Ended Injection Result")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Original: {args.text[:100]}...")
    print("-" * 60)
    print("Injected Document:")
    print(result)


if __name__ == "__main__":
    main()
