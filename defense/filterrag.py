"""
FilterRAG: Threshold-based defense adapted for InceptionRAG.

"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_EPSILON = 0.2
DEFAULT_SIM_THRESHOLD = 0.6

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SLM = "meta-llama/Meta-Llama-3-8B-Instruct"

PROMPT_TEMPLATE = (
    "You are a helpful assistant. The user has provided a query along with "
    "relevant context information. Use this context to answer the question "
    "briefly and clearly. If you cannot find the answer to the question, "
    'respond with "I don\'t know."\n'
    "Contexts: {context}\n"
    "Query: {question}\n"
    "Answer:"
)

_WORD_RE = re.compile(r"[a-z0-9']+")

SlmFn = Callable[[str, str], str]


# ============================================================
# Tokenization
# ============================================================

def tokenize(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


# ============================================================
# Semantic word matcher
# ============================================================

class SemanticWordMatcher:

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self._cache: Dict[str, np.ndarray] = {}

    def _embed(
        self,
        words: Sequence[str],
    ) -> np.ndarray:

        missing = [
            w for w in words
            if w not in self._cache
        ]

        if missing:
            vecs = self.model.encode(
                missing,
                convert_to_numpy=True,
            )

            if not np.isfinite(vecs).all():
                raise ValueError(
                    "FilterRAG: Invalid word embeddings"
                )

            for w, v in zip(missing, vecs):
                self._cache[w] = v

        return np.stack([
            self._cache[w] for w in words
        ])

    def similarity(
        self,
        words_a: Sequence[str],
        words_b: Sequence[str],
    ) -> np.ndarray:

        if not words_a or not words_b:
            return np.zeros(
                (len(words_a), len(words_b))
            )

        a = self._embed(words_a)
        b = self._embed(words_b)

        a = a / np.clip(
            np.linalg.norm(a, axis=1, keepdims=True),
            1e-12,
            None,
        )

        b = b / np.clip(
            np.linalg.norm(b, axis=1, keepdims=True),
            1e-12,
            None,
        )

        return a @ b.T


_MATCHER: Optional[SemanticWordMatcher] = None


def get_matcher() -> SemanticWordMatcher:

    global _MATCHER

    if _MATCHER is None:
        _MATCHER = SemanticWordMatcher()

    return _MATCHER


# ============================================================
# Frequency-Density (Equation 4)
# ============================================================

def freq_density(
    passage_text: str,
    query: str,
    slm_answer: str,
    *,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    matcher: Optional[SemanticWordMatcher] = None,
) -> float:

    doc_counts = Counter(
        tokenize(passage_text)
    )

    keywords = sorted(
        set(tokenize(query))
        | set(tokenize(slm_answer))
    )

    if not doc_counts or not keywords:
        return 0.0

    matcher = matcher or get_matcher()

    doc_words = list(doc_counts)

    sims = matcher.similarity(
        doc_words,
        keywords,
    )

    if not np.isfinite(sims).all():
        raise ValueError(
            "FilterRAG: Invalid similarity scores"
        )

    matched = (
        sims >= sim_threshold
    ).any(axis=1)

    total = sum(
        doc_counts[w]
        for w, m in zip(doc_words, matched)
        if m
    )

    return total / len(doc_counts)


# ============================================================
# Small Language Model (Equation 3)
# ============================================================

def make_slm(
    model_name: str = DEFAULT_SLM,
    max_new_tokens: int = 64,
    device_map: str = "auto",
) -> SlmFn:

    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map=device_map,
    )

    def slm(
        question: str,
        passage: str,
    ) -> str:

        prompt = PROMPT_TEMPLATE.format(
            context=passage,
            question=question,
        )

        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        answer = output[0]["generated_text"].strip()

        if not answer:
            raise RuntimeError(
                "FilterRAG: SLM generated an empty answer"
            )

        return answer

    return slm


# ============================================================
# FilterRAG (Algorithm 1)
# ============================================================

def filterrag(
    query: str,
    retrieved_top_s: Sequence[Dict],
    slm: SlmFn,
    *,
    epsilon: float = DEFAULT_EPSILON,
    top_k: Optional[int] = None,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    matcher: Optional[SemanticWordMatcher] = None,
) -> Tuple[List[Dict], Dict]:

    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("Invalid epsilon")

    if not -1 <= sim_threshold <= 1:
        raise ValueError("Invalid semantic threshold")

    if top_k is not None and top_k < 0:
        raise ValueError("Invalid top_k")

    scores = []
    clean = []

    for idx, d in enumerate(retrieved_top_s):

        # InceptionRAG passage format
        passage_text = d.get("text")

        if not isinstance(passage_text, str):
            raise ValueError(
                f"Invalid passage text at index {idx}"
            )

        # Equation 3: Preliminary answer
        a = slm(query, passage_text)

        if not isinstance(a, str) or not a.strip():
            raise RuntimeError(
                f"Empty SLM answer for passage {idx}"
            )

        # Equation 4: Frequency-Density
        fd = freq_density(
            passage_text,
            query,
            a,
            sim_threshold=sim_threshold,
            matcher=matcher,
        )

        if not np.isfinite(fd):
            raise ValueError(
                "Invalid frequency-density score"
            )

        # Equation 5: Filtering decision
        keep = fd < epsilon

        scores.append({
            "candidate_index": idx,
            "title": d.get("title"),
            "chunk": d.get("chunk"),
            "freq_density_score": fd,
            "slm_answer": a,
            "kept": keep,
        })

        if keep:
            clean.append(d)

    # Select top-k surviving passages
    context = (
        clean[:top_k]
        if top_k is not None
        else clean
    )

    diag = {
        "defense": "filterrag",
        "epsilon": epsilon,
        "semantic_threshold": sim_threshold,
        "top_k": top_k,
        "total_candidates": len(retrieved_top_s),
        "removed_count": len(retrieved_top_s) - len(clean),
        "surviving_count": len(clean),
        "selected_count": len(context),
        "filterrag_scores": scores,
    }

    return context, diag