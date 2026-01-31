"""
InceptionRAG - ZOSO: Zeroth-Order Suffix Optimization

This module implements the ZOSO algorithm for optimizing Anchor and Bridge suffixes
using NTK-GP (Neural Tangent Kernel Gaussian Process) guided search.

Includes:
- DataManager: LLM semantic labeling + Golden Set construction
- RAGEvaluator: Full RAG pipeline (Contriever retrieval + LLM generation)
- NTKGP: Neural Tangent Kernel Gaussian Process for gradient estimation
- ZOSOOptimizer: Two-phase alternating optimization (Anchor/Bridge)

Paper Reference:
    Stage 1: ZOSO (Zeroth-Order Suffix Optimization)
    - Utilizes NTK-GP guided surrogate model
    - Efficiently searches continuous embedding space
    - Identifies authoritative "Anchor" and "Bridge" suffixes
"""

from .zoso_optimizer import (
    ZOSOConfig,
    ZOSOOptimizer,
    SuffixPool,
    ProxyNetwork,
    NTKGP,
    DataManager,
    RAGEvaluator,
)

__all__ = [
    "ZOSOConfig",
    "ZOSOOptimizer",
    "SuffixPool",
    "ProxyNetwork",
    "NTKGP",
    "DataManager",
    "RAGEvaluator",
]
