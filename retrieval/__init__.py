"""
InceptionRAG - Stage 2: Retrieval Hijacking via Dual-Ended Injection

This module implements the dual-ended injection strategy for retrieval hijacking:
1. Subject Header: Prepend query to boost semantic similarity
2. Footer Metadata: Append stealthy keyword tags disguised as metadata
3. Two-Stage Retrieval: First retrieve from corpus, then merge and re-rank with poisoned docs
"""

from .dual_injection import (
    DualInjectionConfig,
    DualEndedInjector,
    TwoStageRetriever,
    RETRIEVAL_TEMPLATES,
    STOPWORDS,
)

__all__ = [
    "DualInjectionConfig",
    "DualEndedInjector",
    "TwoStageRetriever",
    "RETRIEVAL_TEMPLATES",
    "STOPWORDS",
]
