"""
FilterRAG: Threshold-based defense adapted for InceptionRAG.

"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
from dataclasses import dataclass
import yaml


import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_EPSILON = 0.2
DEFAULT_SIM_THRESHOLD = 0.6

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SLM = "Qwen/Qwen2.5-3B-Instruct"

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

        messages = [{"role": "user", "content": prompt}]
        if getattr(pipe.tokenizer, "chat_template", None):
            prompt = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
# Configuration 
# ============================================================

@dataclass
class FilterRAGConfig:
    use_filterrag: bool = False
    epsilon: float = DEFAULT_EPSILON
    semantic_threshold: float = DEFAULT_SIM_THRESHOLD
    slm_model: str = DEFAULT_SLM
    max_new_tokens: int = 64
    embedding_model: str = EMBED_MODEL
    embedding_device: str = "cpu"
    slm_device_map: str = "auto"
    top_k: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "FilterRAGConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = (data.get("defenses", data.get("defense", {})) or {}).get("filterrag", {}) or {}
        return cls(
            use_filterrag=cfg.get("use_filterrag", False),
            epsilon=cfg.get("epsilon", DEFAULT_EPSILON),
            semantic_threshold=cfg.get("semantic_threshold", DEFAULT_SIM_THRESHOLD),
            slm_model=cfg.get("slm_model", DEFAULT_SLM),
            max_new_tokens=cfg.get("max_new_tokens", 64),
            embedding_model=cfg.get("embedding_model", EMBED_MODEL),
            embedding_device=cfg.get("embedding_device", "cpu"),
            slm_device_map=cfg.get("slm_device_map", "auto"),
            top_k=cfg.get("top_k"),
        )


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
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
   
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Invalid epsilon")
    if not np.isfinite(sim_threshold) or not -1 <= sim_threshold <= 1:
        raise ValueError("Invalid semantic threshold")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        raise ValueError("Invalid top_k")

    retained: List[Dict] = []
    rejected: List[Dict] = []
    scores: List[Dict[str, Any]] = []

    for idx, document in enumerate(retrieved_top_s):
        passage_text = document.get("text")
        if not isinstance(passage_text, str):
            raise ValueError(f"Invalid passage text at index {idx}")

        answer = slm(query, passage_text)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"Empty SLM answer for passage {idx}")

        density = freq_density(
            passage_text, query, answer,
            sim_threshold=sim_threshold, matcher=matcher,
        )
        if not np.isfinite(density):
            raise ValueError(f"Invalid frequency-density score at index {idx}")

        keep = bool(density < epsilon)
        scores.append({
            "candidate_index": idx,
            "id": document.get("id"),
            "title": document.get("title"),
            "chunk": document.get("chunk"),
            "doc_type": document.get("doc_type"),
            "freq_density_score": float(density),
            "slm_answer": answer,
            "kept": keep,
        })
        (retained if keep else rejected).append(document)

    selected = retained[:top_k] if top_k is not None else list(retained)
    excluded_by_top_k = retained[len(selected):]
    metadata = {
        "defense": "filterrag",
        "epsilon": float(epsilon),
        "semantic_threshold": float(sim_threshold),
        "top_k": top_k,
        "input_count": len(retrieved_top_s),
        "retained_count": len(retained),
        "rejected_count": len(rejected),
        "selected_count": len(selected),
        "excluded_by_top_k_count": len(excluded_by_top_k),
        "removed_count": len(rejected),
        "surviving_count": len(retained),
        "total_candidates": len(retrieved_top_s),
        "filterrag_scores": scores,
        "selected_context": selected,
        "excluded_by_top_k": excluded_by_top_k,
    }
    return retained, rejected, metadata


class FilterRAGDefense:

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        semantic_threshold: float = DEFAULT_SIM_THRESHOLD,
        slm_model: str = DEFAULT_SLM,
        max_new_tokens: int = 64,
        embedding_model: str = EMBED_MODEL,
        embedding_device: str = "cpu",
        slm_device_map: str = "auto",
        top_k: Optional[int] = None,
        *,
        slm: Optional[SlmFn] = None,
        matcher: Optional[SemanticWordMatcher] = None,
    ):
        self.epsilon = epsilon
        self.semantic_threshold = semantic_threshold
        self.slm_model = slm_model
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.matcher = matcher or SemanticWordMatcher(
            model_name=embedding_model, device=embedding_device
        )
        self.slm = slm or make_slm(
            model_name=slm_model,
            max_new_tokens=max_new_tokens,
            device_map=slm_device_map,
        )

    def apply(
        self, question: str, documents: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
        retained, rejected, metadata = filterrag(
            query=question,
            retrieved_top_s=documents,
            slm=self.slm,
            epsilon=self.epsilon,
            top_k=self.top_k,
            sim_threshold=self.semantic_threshold,
            matcher=self.matcher,
        )
        metadata["slm_model"] = self.slm_model
        metadata["max_new_tokens"] = self.max_new_tokens
        return retained, rejected, metadata
