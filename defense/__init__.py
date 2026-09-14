"""
InceptionRAG Defense Module

HODOR (Heterogeneous Orthogonal Documents to Obstruct Reasoning):
A defense mechanism against multi-document poisoning attacks.
"""

from .hodor import HODORDefense

__all__ = ["HODORDefense"]
