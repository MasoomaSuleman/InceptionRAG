"""
InceptionRAG: Multi-Stage Adversarial Attack on RAG Systems

A sophisticated attack framework that exploits RAG pipelines through:
- Stage 1: Generative Manipulation via Optimized Logic Traps
- Stage 2: Retrieval Hijacking via Dual-Ended Injection  
- Stage 3: Polymorphic Obfuscation and Cascade Test-Time Fine-Tuning (CTTFT)

Usage:
    # Run full attack pipeline
    python -m InceptionRAG.run_attack --help
    
    # Or import components
    from InceptionRAG.generation import LogicTrapGenerator, ZOSOOptimizer
    from InceptionRAG.retrieval import DualEndedInjector, TwoStageRetriever
    from InceptionRAG.llm_client import LLMConfig, create_llm_client
"""

__version__ = "1.0.0"
__author__ = "InceptionRAG Authors"

# LLM Client (supports OpenAI / Azure / Custom API)
from .llm_client import (
    LLMConfig,
    LLMClient,
    create_llm_client,
    get_openai_client,
)

# Stage 1: Generative Manipulation
from .generation import (
    LogicTrapGenerator,
    ZOSOConfig,
    ZOSOOptimizer,
    SuffixPool,
)

# Stage 2: Retrieval Hijacking
from .retrieval import (
    DualInjectionConfig,
    DualEndedInjector,
    TwoStageRetriever,
)

# Stage 3: Full Attack (CTTFT + Polymorphic)
from .run_attack import (
    InceptionRAGConfig,
    InceptionRAGAttacker,
    inject_polymorphic_noise,
)

__all__ = [
    # LLM Client
    "LLMConfig",
    "LLMClient",
    "create_llm_client",
    "get_openai_client",
    # Stage 1
    "LogicTrapGenerator",
    "ZOSOConfig", 
    "ZOSOOptimizer",
    "SuffixPool",
    # Stage 2
    "DualInjectionConfig",
    "DualEndedInjector",
    "TwoStageRetriever",
    # Stage 3
    "InceptionRAGConfig",
    "InceptionRAGAttacker",
    "inject_polymorphic_noise",
]
