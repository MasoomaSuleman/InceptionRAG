"""
InceptionRAG - Stage 1: Generative Manipulation via Optimized Logic Traps

This module implements:
1. Logic Trap generation (Proxy Entity + Attribute/Bridge Documents)
2. ZOSO (Zeroth-Order Suffix Optimization) for Anchor/Bridge suffix optimization
"""

from .logic_trap import (
    LogicTrapGenerator,
    extract_original_entity,
    generate_proxy_entity_and_attribute_doc,
    generate_bridge_document,
)

from .zoso import (
    ZOSOConfig,
    ZOSOOptimizer,
    SuffixPool,
)

__all__ = [
    # Logic Trap
    "LogicTrapGenerator",
    "extract_original_entity",
    "generate_proxy_entity_and_attribute_doc",
    "generate_bridge_document",
    # ZOSO
    "ZOSOConfig",
    "ZOSOOptimizer",
    "SuffixPool",
]
