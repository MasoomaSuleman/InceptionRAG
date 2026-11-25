import re

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
from dataclasses import dataclass
import yaml

import torch
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

SlmFn = Callable[[str, str], str]
_WORD_RE = re.compile(r"[a-z0-9']+")
_MATCHER = None
PROMPT_TEMPLATE = (
    "You are a helpful assistant. The user has provided a query along with relevant context information. "
    "Use this context to answer the question briefly and clearly. "
    "If you cannot find the answer to the question, respond with \'I don\'t know.\' \n"
    "Contexts: {context}\n"
    "Query: {question}\n"
    "Answer:"
)

@dataclass
class FilterRAGConfig:
    use_filterrag: bool = False
    epsilon: float = 0.2
    semantic_threshold: float = 0.6
    max_new_tokens: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    # top_k: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "FilterRAGConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = (data.get("defenses", {})).get("filterrag", {}) 
        llm_config = data.get("llm", {})
        return cls(
            use_filterrag=cfg.get("use_filterrag", False),
            epsilon=cfg.get("epsilon", 0.2),
            semantic_threshold=cfg.get("semantic_threshold", 0.6),
            max_new_tokens=cfg.get("max_new_tokens", 64),
            embedding_model=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_device=cfg.get("embedding_device", "cpu"),
            # top_k=cfg.get("top_k")
        )


class SemanticWordMatcher:
    def __init__(
            self,
            transformer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str = "cpu"
    ):
        self.model = SentenceTransformer(transformer_model_name, device=device)
        self._cache: Dict[str, np.ndarray] = {}

    def _embed(self, words: Sequence[str]) -> np.ndarray:
        missing = [w for w in words if w not in self._cache]

        if missing:
            vecs = self.model.encode(missing, convert_to_numpy=True)

            if not np.isfinite(vecs).all():
                raise ValueError("FilterRAG: Invalid word embeddings")

            for w, v in zip(missing, vecs):
                self._cache[w] = v

        return np.stack([self._cache[w] for w in words])

    def similarity(self, words_a: Sequence[str], words_b: Sequence[str]) -> np.ndarray:
        if not words_a or not words_b:
            return np.zeros((len(words_a), len(words_b)))

        embs = [self._embed(words) for words in [words_a, words_b]]
        embs_norm = [emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None) for emb in embs]

        return embs_norm[0] @ embs_norm[1].T




def tokenize(text: Optional[str]) -> List[str]:
    return _WORD_RE.findall((text or "").lower())

def get_matcher() -> SemanticWordMatcher:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = SemanticWordMatcher()

    return _MATCHER

def freq_density(
    passage_text: str,
    query: str,
    slm_answer: str,
    *,
    sim_threshold: float = 0.6,
    matcher: Optional[SemanticWordMatcher] = None
) -> float:
    doc_counts = Counter(tokenize(passage_text))
    keywords = sorted(set(tokenize(query)) | set(tokenize(slm_answer)))

    if not doc_counts or not keywords:
        return 0.0

    matcher = matcher or get_matcher()

    doc_words = list(doc_counts)
    sims = matcher.similarity(doc_words, keywords)

    if not np.isfinite(sims).all():
        raise ValueError("FilterRAG: Invalid similarity scores")

    matched = (sims >= sim_threshold).any(axis=1)

    total = sum(doc_counts[w] for w, m in zip(doc_words, matched) if m)

    return total / len(doc_counts)

def make_slm(
        model_name: str, 
        max_new_tokens: int = 64,
        device_map: str = "auto"
    ) -> SlmFn:
        pipe = pipeline("text-generation", model=model_name, device_map=device_map)

        def slm(question: str, passage: str) -> str:
            prompt = PROMPT_TEMPLATE.format(context=passage, question=question)

            msgs = [{"role": "user", "content": prompt}]
            if getattr(pipe.tokenizer, "chat_template", None):
                prompt = pipe.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
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
                raise RuntimeError("FilterRAG: SLM generated an empty answer")

            return answer

        return slm




class FilterRAG:
    def __init__(
        self,
        slm,
        tokenizer,
        epsilon: float = 0.2,
        semantic_threshold: float = 0.6,
        max_new_tokens: int = 64,
        embedding_model: str =  "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        # top_k: Optional[int] = None,
        *,
        matcher: Optional[SemanticWordMatcher] = None,
    ):
        self.epsilon = epsilon
        self.semantic_threshold = semantic_threshold
        self.max_new_tokens = max_new_tokens
        # self.top_k = top_k
        self.matcher = matcher or SemanticWordMatcher(embedding_model, embedding_device)
        self.slm = slm 
        self.tokenizer = tokenizer

    def gen_answer(self, question: str, passage: str):
        prompt = PROMPT_TEMPLATE.format(context=passage, question=question)
        
        msgs = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.slm.device)

        with torch.inference_mode():
            output_ids = self.slm.generate( 
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("FilterRAG: SLM generated an empty answer")

        return answer
        

    def apply(self, question: str, documents: List[dict]) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("Invalid epsilon")
        if not np.isfinite(self.semantic_threshold) or not -1 <= self.semantic_threshold <= 1:
            raise ValueError("FilterRAG: Invalid semantic threshold")
        # if self.top_k is not None and (not isinstance(self.top_k, int) or self.top_k < 0):
        #     raise ValueError("FilterRAG: Invalid top_k")
    
        retained: List[Dict] = []
        rejected: List[Dict] = []
        scores: List[Dict[str, Any]] = []
    
        for idx, doc in enumerate(documents):
            passage_text = doc.get("text", "")
            if not isinstance(passage_text, str):
                raise ValueError(f"FilterRAG: Invalid passage text at index {idx}")
    
            answer = self.gen_answer(question, passage_text)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(f"FilterRAG: Empty SLM answer for passage {idx}")
    
            density = freq_density(
                passage_text=passage_text, 
                query=question, 
                slm_answer=answer, 
                sim_threshold=self.semantic_threshold, 
                matcher=self.matcher
            )
            if not np.isfinite(density):
                raise ValueError(f"FilterRAG: Invalid frequency-density score at index {idx}")
    
            keep = bool(density < self.epsilon)
            scores.append({
                "candidate_index": idx,
                "id": doc.get("id"),
                "title": doc.get("title"),
                "chunk": doc.get("chunk"),
                "doc_type": doc.get("doc_type"),
                "freq_density_score": float(density),
                "slm_answer": answer,
                "kept": keep,
            })
    
            (retained if keep else rejected).append(doc)
    
        # selected = retained[:self.top_k] if self.top_k else list(retained)
    
        # excluded_by_top_k = retained[len(selected):]
        metadata = {
            "input_count": len(documents),
            "retained_count": len(retained),
            "rejected_count": len(rejected),
            # "selected_count": len(selected),
            # "excluded_by_top_k_count":len(excluded_by_top_k),
            "filterrag_scores": scores,
            # "selected_context": selected,
            # "excluded_by_top_k": excluded_by_top_k
        }

        return retained, rejected, metadata