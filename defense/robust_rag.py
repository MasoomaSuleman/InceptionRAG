"""
RobustRAG Keyword Aggregation Defense for InceptionRAG.

Adaptation of the KeywordAgg component from:
https://github.com/inspire-group/RobustRAG

Input:
    List[dict] retrieved documents. Each dictionary must contain "text".
    All other metadata is preserved.

Output:
    Final answer generated from aggregated keywords.
    Per-document response records, retaining the original document metadata.

Note:
    This implements keyword aggregation, not decoding aggregation
    or the original formal certification procedure.
"""

import logging
import string
import yaml

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

logger = logging.getLogger(__name__)


@dataclass
class RobustRAGConfig:
    use_robustrag: bool = False

    # Original RobustRAG paper parameters:
    # alpha = relative_threshold
    # beta = absolute_threshold
    relative_threshold: float = 0.2
    absolute_threshold: int = 3

    # Minimum number of non-abstained responses required
    abstention_threshold: int = 1

    max_tokens: int = 100

    @classmethod
    def from_yaml(cls, path: str) -> "RobustRAGConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        defenses_cfg = data.get("defenses", data.get("defense", {}))
        cfg = defenses_cfg.get("robustrag", {})

        return cls(
            use_robustrag=cfg.get("use_robustrag", False),
            relative_threshold=cfg.get("relative_threshold", 0.2),
            absolute_threshold=cfg.get("absolute_threshold", 3),
            abstention_threshold=cfg.get("abstention_threshold", 1),
            max_tokens=cfg.get("max_tokens", 100),
        )


class RobustRAGDefense:

    IGNORE_POS = {
        "VERB", "INTJ", "ADP", "AUX", "CCONJ",
        "DET", "PART", "PRON", "SCONJ",
        "PUNCT", "SPACE",
    }

    def __init__(
        self,
        relative_threshold: float = 0.2,
        absolute_threshold: int = 3,
        abstention_threshold: int = 1,
        max_tokens: int = 100,
    ):
        self.relative = relative_threshold
        self.absolute = absolute_threshold
        self.abstention_threshold = abstention_threshold
        self.max_tokens = max_tokens

        print("[RobustRAG] Loading spaCy model 'en_core_web_sm'...", end=" ", flush=True)

        # Same English NLP model used by the original KeywordAgg.
        self.keyword_extractor = spacy.load("en_core_web_sm")

        print("done")
        print(
            "[RobustRAG] Initialized "
            f"(relative={relative_threshold}, absolute={absolute_threshold}, "
            f"abstention={abstention_threshold}, max_tokens={max_tokens})"
        )

        self.stopwords = set(STOP_WORDS)
        self.punctuation = set(string.punctuation)

    @staticmethod
    def _get_text(document: dict) -> str:
        return str(document.get("text", "")).strip()

    @staticmethod
    def _is_abstention(response: str) -> bool:
        """
        Detect common abstention responses.

        Keep this conservative: don't classify every answer containing
        'don't' as an abstention.
        """
        text = (response or "").strip().lower()

        abstention_patterns = (
            "i don't know",
            "i do not know",
            "i cannot answer",
            "i can't answer",
            "not enough information",
            "insufficient information",
        )

        return any(pattern in text for pattern in abstention_patterns)

    def _chat(
        self,
        llm,
        question: str,
        context: str,
    ) -> str:
        """Use the InceptionRAG LLMClient.chat() interface."""

        return llm.chat(
            user_message=(
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}"
            ),
            system_message=(
                "Answer the question based only on the provided context. "
                "If the context does not contain enough information, "
                "say 'I don't know'. "
                "Output: 'Final answer: <answer>'"
            ),
            temperature=0.0,
            max_tokens=self.max_tokens,
        ).strip()

    def _extract_phrases(self, response: str) -> set:
        """
        Extract unique phrases and individual lemmas.

        This follows the original RobustRAG KeywordAgg approach:
        - retain the complete response
        - split phrases at ignored POS tags
        - add individual lemmas from retained tokens
        - count each phrase only once per response
        """

        doc = self.keyword_extractor(response)

        phrases = {response.strip()}
        current_tokens = []

        def add_current_phrase():
            if not current_tokens:
                return

            phrase = "".join(
                token.lemma_ + token.whitespace_
                for token in current_tokens
            ).strip()

            if phrase:
                phrases.add(phrase)

            for token in current_tokens:
                lemma = token.lemma_.strip()
                if lemma:
                    phrases.add(lemma)

        for token in doc:
            if token.pos_ in self.IGNORE_POS:
                add_current_phrase()
                current_tokens = []
            else:
                current_tokens.append(token)

        add_current_phrase()

        return phrases

    def _aggregate_keywords(
        self,
        individual_results: List[dict],
    ) -> Tuple[List[str], Dict[str, int], float]:
        """
        Aggregate keywords across non-abstained responses.

        A phrase is retained if its frequency is at least:

            min(absolute_threshold,
                relative_threshold * number_of_non_abstained_responses)
        """

        response_phrase_sets = []

        for record in individual_results:
            if record["abstained"]:
                continue

            response = record["response"]

            if not response:
                continue

            response_phrase_sets.append(
                self._extract_phrases(response)
            )

        if not response_phrase_sets:
            return [], {}, 0.0

        token_counter = Counter()

        for phrase_set in response_phrase_sets:
            for phrase in phrase_set:
                token_counter[phrase] += 1

        threshold = min(
            self.absolute,
            self.relative * len(response_phrase_sets),
        )

        for phrase, count in list(token_counter.items()):
            normalized = phrase.strip().lower()

            if count < threshold:
                del token_counter[phrase]
                continue

            if normalized in self.punctuation:
                del token_counter[phrase]
                continue

            if normalized in self.stopwords:
                del token_counter[phrase]
                continue

        # Same ordering principle as original RobustRAG:
        # longer phrases first, then alphabetical order.
        sorted_phrases = sorted(
            token_counter.items(),
            key=lambda item: (len(item[0]), item[0]),
            reverse=True,
        )

        retained_keywords = [
            phrase for phrase, _ in sorted_phrases
        ]

        return (
            retained_keywords,
            dict(token_counter),
            threshold,
        )

    def _generate_final_answer(
        self,
        question: str,
        keywords: List[str],
        llm,
    ) -> str:
        """Generate the final answer from aggregated keyword hints."""

        hints = ", ".join(keywords)

        return llm.chat(
            user_message=(
                f"QUESTION: {question}\n\n"
                f"Aggregated information from independently retrieved "
                f"documents:\n{hints}\n\n"
                "Use the aggregated information above to answer the "
                "question. Do not invent unsupported facts. "
                "Output: 'Final answer: <answer>'"
            ),
            system_message=(
                "You are answering a question using information "
                "aggregated from multiple independent document responses. "
                "Use only the supplied information. Be concise."
            ),
            temperature=0.0,
            max_tokens=self.max_tokens,
        ).strip()

    def apply(
        self,
        question: str,
        documents: List[dict],
        llm,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply RobustRAG Keyword Aggregation.

        Each individual response is linked to its original document.
        Original document metadata is preserved.
        """

        defense_info = {
            "defense_type": "robustrag_keyword_aggregation",
            "num_input_documents": len(documents),
            "individual_responses": [],
            "abstained_indices": [],
            "num_non_abstained_responses": 0,
            "keyword_frequencies": {},
            "frequency_threshold": None,
            "retained_keywords": [],
            "num_retained_keywords": 0,
            "final_answer": "",
            "certification_performed": False,
        }

        print(f"\n[RobustRAG] Starting keyword aggregation for {len(documents)} document(s)")

        if not documents:
            answer = "I don't have enough information to answer."
            defense_info["final_answer"] = answer
            print("[RobustRAG] No documents received; returning a fallback answer")
            return answer, defense_info

        individual_results = []

        # --------------------------------------------------
        # Step 1: Generate one response per retrieved record
        # --------------------------------------------------

        for idx, document in enumerate(documents):
            print(
                f"[RobustRAG] [{idx + 1}/{len(documents)}] "
                f"Processing document {idx + 1}...",
                end=" ",
                flush=True,
            )
            document_text = self._get_text(document)

            if not document_text:
                individual_results.append({
                    "document_index": idx,
                    "document": dict(document),
                    "response": "",
                    "abstained": True,
                    "status": "skipped_empty_text",
                })
                print("skipped (empty text)")
                continue

            try:
                response = self._chat(
                    llm=llm,
                    question=question,
                    context=document_text,
                )

                abstained = self._is_abstention(response)

                record = {
                    "document_index": idx,
                    "document": dict(document),
                    "response": response,
                    "abstained": abstained,
                    "status": "completed",
                }

                print("abstained" if abstained else "answered")

            except Exception as exc:
                logger.exception(
                    "RobustRAG failed on document index %s", idx
                )

                record = {
                    "document_index": idx,
                    "document": dict(document),
                    "response": "",
                    "abstained": True,
                    "status": "error",
                    "error": str(exc),
                }
                print(f"error: {exc}")

            individual_results.append(record)

        defense_info["individual_responses"] = individual_results

        defense_info["abstained_indices"] = [
            record["document_index"]
            for record in individual_results
            if record["abstained"]
        ]

        non_abstained = [
            record for record in individual_results
            if not record["abstained"] and record["response"]
        ]

        defense_info["num_non_abstained_responses"] = len(non_abstained)

        print(
            f"[RobustRAG] Document processing complete: "
            f"{len(non_abstained)} answered, "
            f"{len(individual_results) - len(non_abstained)} abstained/error"
        )

        # --------------------------------------------------
        # Step 2: Check abstention threshold
        # --------------------------------------------------

        if len(non_abstained) < self.abstention_threshold:
            answer = "I don't know."

            defense_info["abstained"] = True
            defense_info["final_answer"] = answer

            return answer, defense_info

        # --------------------------------------------------
        # Step 3: Aggregate keywords
        # --------------------------------------------------

        keywords, frequencies, threshold = self._aggregate_keywords(
            individual_results
        )

        defense_info["keyword_frequencies"] = frequencies
        defense_info["frequency_threshold"] = threshold
        defense_info["retained_keywords"] = keywords
        defense_info["num_retained_keywords"] = len(keywords)

        print(
            f"[RobustRAG] Aggregated {len(frequencies)} retained phrases "
            f"with threshold {threshold:.2f}"
        )
        if keywords:
            preview = ", ".join(keywords[:10])
            print(f"[RobustRAG] Retained keywords preview: {preview}")

        # --------------------------------------------------
        # Step 4: Generate final answer from aggregated hints
        # --------------------------------------------------

        if not keywords:
            answer = "I don't know."
            defense_info["abstained"] = True
        else:
            answer = self._generate_final_answer(
                question=question,
                keywords=keywords,
                llm=llm,
            )
            defense_info["abstained"] = False

        defense_info["final_answer"] = answer

        return answer, defense_info
