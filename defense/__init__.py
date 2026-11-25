"""
InceptionRAG Defense Module

HODOR (Heterogeneous Orthogonal Documents to Obstruct Reasoning):
A defense mechanism against multi-document poisoning attacks.
"""

from .hodor import HODORDefense, HODORConfig
from .trust_rag import TrustRAGDefense, TrustRAGConfig
from .robust_rag import RobustRAGDefense, RobustRAGConfig
from .av_filter import AVFilter, AVFilterConfig, calibrate_threshold, init_score_fn
from .filterrag import FilterRAG, FilterRAGConfig

__all__ = [
    "HODORDefense",
    "HODORConfig",
    "TrustRAGDefense",
    "TrustRAGConfig",
    "RobustRAGDefense",
    "RobustRAGConfig",
    "AVFilter",
    "AVFilterConfig",
    "calibrate_threshold"
    "init_score_fn"
    "FilterRAG",
    "FilterRAGConfig"
]

