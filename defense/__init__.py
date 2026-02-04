"""
InceptionRAG Defense Module

HODOR (Holistic Document-wise Outcome Reasoning):
A defense mechanism against multi-document poisoning attacks.
"""

from .hodor import HODORDefense

__all__ = ["HODORDefense"]
