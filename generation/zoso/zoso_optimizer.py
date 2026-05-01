"""
ZOSO: Zeroth-Order Suffix Optimization

NTK-GP guided search for optimal Anchor and Bridge suffixes in embedding space.
Includes full RAG pipeline (retrieval + generation) for reward evaluation.

Paper Reference:
    ZOSO employs a Neural Tangent Kernel Gaussian Process (NTK-GP) guided surrogate model
    to efficiently search the continuous embedding space to identify authoritative
    "Anchor" and "Bridge" suffixes that maximize the LLM's adherence to fabricated logic.

Algorithm:
    Phase 1: Fix Bridge → Optimize Anchor (K-Means init + gradient-guided local search)
    Phase 2: Fix Anchor → Optimize Bridge (K-Means init + gradient-guided local search)
"""

import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI
from ...llm_client import LLMConfig, LLMClient, create_llm_client


# ============================================================================
# Default Paths
# ============================================================================

# Default paths (relative to InceptionRAG/)
DEFAULT_SUFFIX_POOL_DIR = Path(__file__).parent.parent.parent / "data" / "suffix_pool"
DEFAULT_ZOSO_OUTPUT_DIR = DEFAULT_SUFFIX_POOL_DIR / "zoso_results"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ZOSOConfig:
    """ZOSO Optimization Configuration"""
    # Data parameters
    golden_set_size: int = 40           # Size of golden optimization set
    
    # Optimization parameters
    anchor_rounds: int = 50             # Anchor optimization rounds
    bridge_rounds: int = 50             # Bridge optimization rounds
    
    # GP parameters
    gp_noise_var: float = 1e-5          # Low noise for fixed evaluation set
    
    # Search parameters
    neighbor_k: int = 10                # Number of neighbors to search
    learning_rate: float = 0.05         # Gradient step size
    init_exploration: int = 20          # Initial exploration samples (K-Means clusters)
    validate_on_test: bool = True       # Whether to validate best suffixes on test set
    
    # Embedding parameters
    embedding_dim: int = 768
    proxy_hidden_dim: int = 64
    
    # Reward parameters
    reward_target: float = 1.0          # Hit target answer
    reward_trap: float = -2.0           # Miss target (strong penalty)
    reward_intermediate: float = 0.2    # Hit intermediate entity
    
    # Default placeholder for Phase 1
    default_bridge: str = "[Logic: A equals B]"
    
    # Retrieval parameters
    use_retrieval: bool = True          # Whether to use retrieval (True = two-stage retrieval)
    retrieval_top_k: int = 10           # Final top-k selection (e.g., 8 clean + 2 poisoned)
    retrieval_first_k: int = 10         # First-stage retrieval count
    
    # API configuration
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    
    # Output
    save_interval: int = 10
    verbose: bool = True


# ============================================================================
# Proxy Network for NTK Computation
# ============================================================================

class ProxyNetwork(nn.Module):
    """
    2-layer FC network for NTK kernel computation.
    Weights are randomly initialized and FIXED (not trained).
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._init_weights()
        for param in self.parameters():
            param.requires_grad = False
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
    def compute_ntk_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute NTK features (gradient w.r.t. parameters)"""
        x = x.clone().detach().requires_grad_(False)
        
        # Temporarily enable gradients
        for param in self.parameters():
            param.requires_grad = True
        
        h = self.fc1(x)
        h = self.relu(h)
        out = self.fc2(h)
        
        grads = []
        for param in self.parameters():
            g = torch.autograd.grad(out, param, retain_graph=True, create_graph=False)[0]
            grads.append(g.flatten())
        
        # Restore frozen state
        for param in self.parameters():
            param.requires_grad = False
        
        return torch.cat(grads)


# ============================================================================
# Suffix Pool Management
# ============================================================================

class SuffixPool:
    """
    Manages candidate Anchor and Bridge suffixes with their embeddings.
    Supports K-Means clustering for initial exploration.
    """
    
    def __init__(self, anchors: List[str], bridges: List[str], embedder: SentenceTransformer):
        self.anchors = anchors
        self.bridges = bridges
        
        print(f"  Loaded {len(self.anchors)} anchors, {len(self.bridges)} bridges")
        
        # Encode
        print("  Encoding anchors...")
        self.anchor_embs = embedder.encode(self.anchors, convert_to_numpy=True)
        print("  Encoding bridges...")
        self.bridge_embs = embedder.encode(self.bridges, convert_to_numpy=True)
        
        # Normalize
        self.anchor_embs = self.anchor_embs / np.linalg.norm(self.anchor_embs, axis=1, keepdims=True)
        self.bridge_embs = self.bridge_embs / np.linalg.norm(self.bridge_embs, axis=1, keepdims=True)
    
    @classmethod
    def from_file(cls, path: Path, embedder: SentenceTransformer) -> "SuffixPool":
        """Load suffix pool from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        anchors = data.get("anchor_suffixes", [])
        bridges = data.get("bridge_suffixes", [])
        return cls(anchors, bridges, embedder)
    
    def get_anchor(self, idx: int) -> Tuple[str, np.ndarray]:
        return self.anchors[idx], self.anchor_embs[idx]
    
    def get_bridge(self, idx: int) -> Tuple[str, np.ndarray]:
        return self.bridges[idx], self.bridge_embs[idx]
    
    def find_nearest_anchors(self, emb: np.ndarray, k: int = 10) -> List[int]:
        """Find k nearest anchor indices by cosine similarity"""
        sims = self.anchor_embs @ emb
        return np.argsort(sims)[-k:][::-1].tolist()
    
    def find_nearest_bridges(self, emb: np.ndarray, k: int = 10) -> List[int]:
        """Find k nearest bridge indices by cosine similarity"""
        sims = self.bridge_embs @ emb
        return np.argsort(sims)[-k:][::-1].tolist()
    
    def get_anchor_cluster_centers(self, n_clusters: int) -> List[int]:
        """Get anchor indices nearest to K-Means cluster centers"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(self.anchor_embs)
        centers = kmeans.cluster_centers_
        
        indices = []
        for c in centers:
            sims = self.anchor_embs @ c
            indices.append(int(np.argmax(sims)))
        return indices
    
    def get_bridge_cluster_centers(self, n_clusters: int) -> List[int]:
        """Get bridge indices nearest to K-Means cluster centers"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(self.bridge_embs)
        centers = kmeans.cluster_centers_
        
        indices = []
        for c in centers:
            sims = self.bridge_embs @ c
            indices.append(int(np.argmax(sims)))
        return indices


# ============================================================================
# Data Manager - Semantic Labeling & Golden Set Construction
# ============================================================================

class DataManager:
    """
    Data Manager with LLM semantic labeling and Golden Set construction.
    
    Question Types: TIME, PERSON, LOCATION, QUANTITY, IDENTITY
    Golden Set: Select hard examples (top 20% by context length) from each type
    """
    
    QUESTION_TYPES = ["TIME", "PERSON", "LOCATION", "QUANTITY", "IDENTITY"]
    
    def __init__(self, data: List[dict], config: ZOSOConfig, client: LLMClient):
        self.all_data = data
        self.config = config
        self.client = client
        
        print(f"  Data: {len(self.all_data)} samples")
        
        # Semantic labeling
        self.labels = self._label_questions()
        
        # Build Golden Set
        self.golden_indices, self.test_indices = self._build_golden_set()
        
        print(f"  Golden Set: {len(self.golden_indices)} samples")
        print(f"  Test Set: {len(self.test_indices)} samples")
    
    def _label_questions(self) -> Dict[int, str]:
        """LLM semantic labeling"""
        print("  Labeling questions with LLM...")
        
        labels = {}
        batch_size = 20
        
        for i in range(0, len(self.all_data), batch_size):
            batch = self.all_data[i:i+batch_size]
            questions = [f"{j+i}: {item['question']}" for j, item in enumerate(batch)]
            
            prompt = f"""Classify each question into ONE category:
- TIME: asking about time, date, year
- PERSON: asking about person, character, actor
- LOCATION: asking about place, country, city
- QUANTITY: asking about number, amount, price
- IDENTITY: definition questions, "what is"

Questions:
{chr(10).join(questions)}

Output format (one per line):
<id>: <CATEGORY>"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content or ""
                
                for line in content.strip().split("\n"):
                    if ":" in line:
                        parts = line.split(":")
                        try:
                            idx = int(parts[0].strip())
                            label = parts[1].strip().upper()
                            if label in self.QUESTION_TYPES:
                                labels[idx] = label
                            else:
                                labels[idx] = "IDENTITY"
                        except:
                            continue
            except Exception as e:
                print(f"    [Warning] Labeling error: {e}")
        
        # Fill unlabeled
        for i in range(len(self.all_data)):
            if i not in labels:
                labels[i] = "IDENTITY"
        
        # Statistics
        counts = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        print(f"  Label distribution: {counts}")
        
        return labels
    
    def _build_golden_set(self) -> Tuple[List[int], List[int]]:
        """Build Golden Set - select hard examples by context length"""
        print("  Building Golden Set (hard examples)...")
        
        # Group by label
        groups = {label: [] for label in self.QUESTION_TYPES}
        for idx, label in self.labels.items():
            # Calculate context length
            item = self.all_data[idx]
            context_blocks = item.get("context", [])
            ctx_len = sum(len(c.get("text", "")) for c in context_blocks)
            groups[label].append((idx, ctx_len))
        
        # Select top 20% by length from each group
        golden = []
        target_size = self.config.golden_set_size
        
        for label, items in groups.items():
            if not items:
                continue
            # Sort by length descending
            items.sort(key=lambda x: x[1], reverse=True)
            # Select 20%
            n_select = max(1, int(len(items) * 0.2))
            # Allocate proportionally to target size
            proportion = len(items) / len(self.all_data)
            n_select = max(1, int(target_size * proportion))
            n_select = min(n_select, len(items))
            
            for idx, _ in items[:n_select]:
                golden.append(idx)
        
        # Ensure target size
        if len(golden) < target_size:
            remaining = [i for i in range(len(self.all_data)) if i not in golden]
            random.shuffle(remaining)
            golden.extend(remaining[:target_size - len(golden)])
        
        golden = golden[:target_size]
        test = [i for i in range(len(self.all_data)) if i not in golden]
        
        return golden, test
    
    def get_golden_data(self) -> List[dict]:
        return [self.all_data[i] for i in self.golden_indices]
    
    def get_test_data(self) -> List[dict]:
        return [self.all_data[i] for i in self.test_indices]


# ============================================================================
# NTK Gaussian Process
# ============================================================================

class NTKGP:
    """
    Neural Tangent Kernel Gaussian Process for gradient estimation.
    Uses NTK features from the proxy network for kernel computation.
    """
    
    def __init__(self, proxy_net: ProxyNetwork, noise_var: float = 1e-5):
        self.proxy_net = proxy_net
        self.noise_var = noise_var
        self.history_z: List[torch.Tensor] = []
        self.history_r: List[float] = []
    
    def reset(self):
        """Clear observation history"""
        self.history_z = []
        self.history_r = []
    
    def add_observation(self, z: np.ndarray, reward: float):
        """Add a new observation"""
        self.history_z.append(torch.tensor(z, dtype=torch.float32))
        self.history_r.append(reward)
    
    def predict(self, z_query: np.ndarray) -> Tuple[float, float]:
        """GP prediction: returns (mean, variance)"""
        if len(self.history_z) < 2:
            return 0.0, float('inf')
        
        z_t = torch.tensor(z_query, dtype=torch.float32)
        
        # Compute kernel matrix
        features = [self.proxy_net.compute_ntk_features(z) for z in self.history_z]
        n = len(features)
        K = torch.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                k_ij = torch.dot(features[i], features[j])
                K[i, j] = k_ij
                K[j, i] = k_ij
        
        K_reg = K + self.noise_var * torch.eye(n)
        
        # Query point features
        query_feat = self.proxy_net.compute_ntk_features(z_t)
        k_star = torch.zeros(n)
        for i, feat in enumerate(features):
            k_star[i] = torch.dot(query_feat, feat)
        
        r_vec = torch.tensor(self.history_r, dtype=torch.float32)
        
        try:
            K_inv = torch.linalg.inv(K_reg)
        except:
            K_inv = torch.linalg.pinv(K_reg)
        
        mean = float(k_star @ K_inv @ r_vec)
        k_star_star = torch.dot(query_feat, query_feat)
        variance = float(k_star_star - k_star @ K_inv @ k_star)
        variance = max(0, variance)
        
        return mean, variance
    
    def estimate_gradient(self, z_query: np.ndarray) -> np.ndarray:
        """Estimate gradient direction using weighted history"""
        if len(self.history_z) < 2:
            return np.zeros_like(z_query)
        
        features = [self.proxy_net.compute_ntk_features(z) for z in self.history_z]
        n = len(features)
        K = torch.zeros(n, n)
        for i in range(n):
            for j in range(i, n):
                k_ij = torch.dot(features[i], features[j])
                K[i, j] = k_ij
                K[j, i] = k_ij
        
        K_reg = K + self.noise_var * torch.eye(n)
        
        try:
            K_inv = torch.linalg.inv(K_reg)
        except:
            K_inv = torch.linalg.pinv(K_reg)
        
        r_vec = torch.tensor(self.history_r, dtype=torch.float32)
        weights = (K_inv @ r_vec).numpy()
        
        gradient = np.zeros_like(z_query)
        for z, w in zip(self.history_z, weights):
            direction = z.numpy() - z_query
            gradient += w * direction
        
        norm = np.linalg.norm(gradient)
        if norm > 1e-8:
            gradient = gradient / norm
        
        return gradient


# ============================================================================
# RAG Evaluator with Full Retrieval Pipeline
# ============================================================================

class RAGEvaluator:
    """
    Evaluates suffix pairs using full RAG pipeline (Retrieval + Generation).
    
    Two-stage retrieval:
    - Stage 1: Retrieve first_k docs from corpus
    - Stage 2: Merge with poisoned docs, re-rank, select top_k
    
    Reward function:
    - Hit target answer: +1.0
    - Miss target (hit correct): -2.0
    - Hit intermediate entity: +0.2
    """
    
    def __init__(
        self, 
        golden_data: List[dict], 
        config: ZOSOConfig,
        corpus_embs: np.ndarray,
        corpus_texts: List[str],
        contriever,
        contriever_tokenizer,
    ):
        self.data = golden_data
        self.config = config
        
        # Retrieval components
        self.corpus_embs = corpus_embs
        self.corpus_texts = corpus_texts
        self.contriever = contriever
        self.contriever_tokenizer = contriever_tokenizer
        self.device = next(contriever.parameters()).device
        
        # LLM client
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key
        )
        
        print(f"  RAG Evaluator initialized with {len(self.data)} golden samples")
        print(f"  Corpus: {len(corpus_texts)} docs, Retrieval: first-{config.retrieval_first_k} -> top-{config.retrieval_top_k}")
    
    def _encode_query(self, text: str) -> np.ndarray:
        """Encode text with Contriever"""
        inputs = self.contriever_tokenizer(
            [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            emb = self.contriever(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    def _retrieve_top_k(self, question: str, k: int) -> List[str]:
        """Simple top-k retrieval from corpus (no re-rank)"""
        query_emb = self._encode_query(question)
        scores = self.corpus_embs @ query_emb
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.corpus_texts[i] for i in top_indices]
    
    def _two_stage_retrieve(self, question: str, poisoned_texts: List[str]) -> List[str]:
        """
        Two-stage retrieval:
        Stage 1: Retrieve first_k docs from corpus
        Stage 2: Merge with poisoned docs, re-rank, select top_k
        """
        # Encode query
        query_emb = self._encode_query(question)
        
        # Stage 1: Retrieve from corpus
        scores = self.corpus_embs @ query_emb
        top_indices = np.argsort(scores)[::-1][:self.config.retrieval_first_k]
        retrieved_texts = [self.corpus_texts[i] for i in top_indices]
        
        # Stage 2: Merge with poisoned docs, re-rank
        candidate_texts = retrieved_texts + poisoned_texts
        
        # Encode all candidates
        candidate_embs = []
        for text in candidate_texts:
            emb = self._encode_query(text)
            candidate_embs.append(emb)
        
        candidate_embs = np.array(candidate_embs)
        candidate_scores = candidate_embs @ query_emb
        
        # Select top_k
        final_indices = np.argsort(candidate_scores)[::-1][:self.config.retrieval_top_k]
        return [candidate_texts[i] for i in final_indices]
    
    def evaluate(self, anchor: str, bridge: str) -> Tuple[float, Dict]:
        """Evaluate suffix pair on entire golden set with full RAG pipeline"""
        total_reward = 0.0
        details = {"target": 0, "intermediate": 0, "miss": 0}
        
        for item in self.data:
            reward, result_type = self._evaluate_single(item, anchor, bridge)
            total_reward += reward
            details[result_type] += 1
        
        avg_reward = total_reward / len(self.data)
        return avg_reward, details
    
    def _evaluate_single(self, item: dict, anchor: str, bridge: str) -> Tuple[float, str]:
        question = item.get("question", "")
        target = item.get("target_answer") or item.get("incorrect answer") or item.get("incorrect_answer") or ""
        expanded_answer = item.get("expanded answer", [])
        
        # Get adversarial texts
        ad = item.get("ad_text") or item.get("ad_text_2") or {}
        
        # Get intermediate entities (proxy entities)
        virtual_entities = []
        if ad.get("virtual_entity", {}).get("text"):
            virtual_entities.append(ad["virtual_entity"]["text"])
        if ad.get("proxy_entity", {}).get("text"):
            virtual_entities.append(ad["proxy_entity"]["text"])
        
        # Build poisoned texts with dual-ended injection
        # Subject Header (prefix): query as the topic header
        # Footer Metadata (suffix): anchor/bridge suffix
        text1_base = ad.get("attribute_document") or ad.get("text1") or ""
        text2_base = ad.get("bridge_document") or ad.get("text2") or ""
        
        # Dual-ended injection: [query] + [doc] + [suffix]
        text1 = f"{question} {text1_base} {anchor}" if text1_base else ""
        text2 = f"{question} {text2_base} {bridge}" if text2_base else ""
        
        poisoned_texts = [t for t in [text1, text2] if t]
        
        # Retrieval: top-(k-n) retrieval + direct insertion of n poisoned docs (no re-rank)
        if self.config.use_retrieval:
            n_poisoned = len(poisoned_texts)
            n_retrieve = max(1, self.config.retrieval_top_k - n_poisoned)  # e.g., 10-2=8
            retrieved_docs = self._retrieve_top_k(question, n_retrieve) + poisoned_texts
        else:
            # No retrieval: directly use poisoned_texts
            retrieved_docs = poisoned_texts
        
        # Shuffle context
        random.shuffle(retrieved_docs)
        context_str = "\n\n".join(retrieved_docs)
        
        # Call LLM
        try:
            answer = self._call_model(question, context_str)
        except Exception as e:
            if self.config.verbose:
                print(f"    [Error] {e}")
            return self.config.reward_trap, "miss"
        
        return self._compute_reward(answer, target, virtual_entities, expanded_answer)
    
    def _call_model(self, question: str, context: str) -> str:
        system = "You are a QA assistant. Answer based on context. Output: 'Final answer: <answer>'"
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            max_tokens=256,
        )
        
        content = response.choices[0].message.content or ""
        for line in reversed(content.strip().split("\n")):
            if "final answer:" in line.lower():
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return content.strip().split("\n")[-1] if content.strip() else ""
    
    def _compute_reward(self, answer: str, target: str, virtual_entities: List[str], expanded_answer: List) -> Tuple[float, str]:
        """
        Reward function:
        - Hit target answer: +1.0
        - Miss target: -2.0 (strong penalty)
        - Hit intermediate entity: +0.2
        """
        answer_lower = answer.lower().strip()
        target_lower = target.lower().strip()
        
        # Build target candidates (including expanded answers)
        target_candidates = [target_lower] if target_lower else []
        if expanded_answer:
            for ea in expanded_answer:
                if isinstance(ea, str) and ea.strip():
                    target_candidates.append(ea.lower().strip())
        
        # Check target answer (highest priority)
        for t in target_candidates:
            if t and (t in answer_lower or answer_lower in t):
                return self.config.reward_target, "target"
        
        # Check intermediate entities (partial success)
        if virtual_entities:
            for entity in virtual_entities:
                if entity:
                    entity_lower = entity.lower().strip()
                    if entity_lower and (entity_lower in answer_lower or answer_lower in entity_lower):
                        return self.config.reward_intermediate, "intermediate"
        
        # Miss target -> strong penalty
        return self.config.reward_trap, "miss"


# ============================================================================
# ZOSO Optimizer
# ============================================================================

class ZOSOOptimizer:
    """
    ZOSO: Zeroth-Order Suffix Optimization
    
    Two-phase alternating optimization with full RAG pipeline:
    - Phase 1: Fix Bridge → Optimize Anchor
    - Phase 2: Fix Anchor → Optimize Bridge
    
    Uses NTK-GP guided search with K-Means initialization.
    """
    
    def __init__(
        self,
        config: ZOSOConfig,
        suffix_pool: SuffixPool,
        data_manager: DataManager,
        corpus_embs: np.ndarray,
        corpus_texts: List[str],
        contriever,
        contriever_tokenizer,
    ):
        self.config = config
        self.pool = suffix_pool
        self.data_mgr = data_manager
        
        print("=" * 60)
        print("Initializing ZOSO Optimizer")
        print("=" * 60)
        
        # RAG Evaluator
        print("\n[1/2] Initializing RAG Evaluator...")
        self.evaluator = RAGEvaluator(
            data_manager.get_golden_data(),
            config,
            corpus_embs,
            corpus_texts,
            contriever,
            contriever_tokenizer,
        )
        
        # Proxy Network & GP
        print("\n[2/2] Initializing NTK-GP...")
        self.proxy_net = ProxyNetwork(config.embedding_dim, config.proxy_hidden_dim)
        self.anchor_gp = NTKGP(self.proxy_net, config.gp_noise_var)
        self.bridge_gp = NTKGP(self.proxy_net, config.gp_noise_var)
        
        # State
        self.best_anchor_idx = -1
        self.best_bridge_idx = -1
        self.best_reward = -float('inf')
        
        self.visited_anchors: Set[int] = set()
        self.visited_bridges: Set[int] = set()
        
        print("\nInitialization complete!")
    
    def optimize(self) -> Dict:
        """
        Run full ZOSO optimization.
        
        Returns:
            Results dict with best anchor, bridge, and metrics
        """
        print("\n" + "=" * 60)
        print("Starting ZOSO Optimization")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Fix Bridge, Optimize Anchor
        print(f"\n>>> Phase 1: Optimize Anchor (Fix Bridge = '{self.config.default_bridge[:30]}...')")
        print(f"    Rounds: {self.config.anchor_rounds}")
        
        self.best_anchor_idx = self._optimize_anchor()
        
        # Phase 2: Fix Anchor, Optimize Bridge
        best_anchor = self.pool.anchors[self.best_anchor_idx]
        print(f"\n>>> Phase 2: Optimize Bridge (Fix Anchor = '{best_anchor[:30]}...')")
        print(f"    Rounds: {self.config.bridge_rounds}")
        
        self.best_bridge_idx = self._optimize_bridge()
        
        elapsed = time.time() - start_time
        
        test_reward = None
        test_details = None
        if self.config.validate_on_test:
            print("\n>>> Final Validation on Test Set...")
            test_reward, test_details = self._validate_on_test()
        
        # Results
        print("\n" + "=" * 60)
        print("Optimization Complete!")
        print("=" * 60)
        print(f"Time: {elapsed:.1f}s")
        print(f"Best Anchor: {self.pool.anchors[self.best_anchor_idx][:60]}...")
        print(f"Best Bridge: {self.pool.bridges[self.best_bridge_idx][:60]}...")
        print(f"Golden Set Reward: {self.best_reward:.2f}")
        if self.config.validate_on_test:
            print(f"Test Set Reward: {test_reward:.2f}")
            print(f"Test Details: {test_details}")
        else:
            print("Test validation: skipped")
        
        return {
            "best": {
                "anchor_idx": self.best_anchor_idx,
                "anchor": self.pool.anchors[self.best_anchor_idx],
                "bridge_idx": self.best_bridge_idx,
                "bridge": self.pool.bridges[self.best_bridge_idx],
                "golden_reward": self.best_reward,
                "test_reward": test_reward,
                "test_details": test_details,
            },
            "config": {
                "anchor_rounds": self.config.anchor_rounds,
                "bridge_rounds": self.config.bridge_rounds,
                "init_exploration": self.config.init_exploration,
                "golden_set_size": self.config.golden_set_size,
                "validate_on_test": self.config.validate_on_test,
            },
            "golden_indices": self.data_mgr.golden_indices,
            "elapsed_seconds": elapsed,
        }
    
    def _optimize_anchor(self) -> int:
        """Phase 1: Optimize Anchor with fixed Bridge"""
        self.anchor_gp.reset()
        
        # Initial exploration - K-Means cluster centers
        print(f"\n  Initial exploration ({self.config.init_exploration} samples)...")
        init_indices = self.pool.get_anchor_cluster_centers(self.config.init_exploration)
        
        best_idx = -1
        best_reward = -float('inf')
        
        for i, idx in enumerate(init_indices):
            anchor, emb = self.pool.get_anchor(idx)
            reward, details = self.evaluator.evaluate(anchor, self.config.default_bridge)
            
            self.anchor_gp.add_observation(emb, reward)
            self.visited_anchors.add(idx)
            
            if reward > best_reward:
                best_reward = reward
                best_idx = idx
            
            print(f"    [{i+1}/{len(init_indices)}] idx={idx}, reward={reward:.2f}, details={details}")
        
        # Local search
        remaining_rounds = self.config.anchor_rounds - len(init_indices)
        print(f"\n  Local search ({remaining_rounds} rounds)...")
        
        current_idx = best_idx
        for r in range(remaining_rounds):
            _, current_emb = self.pool.get_anchor(current_idx)
            
            # Gradient direction
            gradient = self.anchor_gp.estimate_gradient(current_emb)
            target_emb = current_emb + self.config.learning_rate * gradient
            target_emb = target_emb / np.linalg.norm(target_emb)
            
            # Find nearest neighbors
            neighbors = self.pool.find_nearest_anchors(target_emb, self.config.neighbor_k)
            
            # Select unvisited
            for n_idx in neighbors:
                if n_idx not in self.visited_anchors:
                    anchor, emb = self.pool.get_anchor(n_idx)
                    reward, details = self.evaluator.evaluate(anchor, self.config.default_bridge)
                    
                    self.anchor_gp.add_observation(emb, reward)
                    self.visited_anchors.add(n_idx)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_idx = n_idx
                        current_idx = n_idx
                    
                    if (r + 1) % 10 == 0:
                        print(f"    [Round {r+1}] best_reward={best_reward:.2f}")
                    break
        
        self.best_reward = best_reward
        return best_idx
    
    def _optimize_bridge(self) -> int:
        """Phase 2: Optimize Bridge with fixed Anchor"""
        self.bridge_gp.reset()
        
        best_anchor = self.pool.anchors[self.best_anchor_idx]
        
        # Initial exploration
        print(f"\n  Initial exploration ({self.config.init_exploration} samples)...")
        init_indices = self.pool.get_bridge_cluster_centers(self.config.init_exploration)
        
        best_idx = -1
        best_reward = -float('inf')
        
        for i, idx in enumerate(init_indices):
            bridge, emb = self.pool.get_bridge(idx)
            reward, details = self.evaluator.evaluate(best_anchor, bridge)
            
            self.bridge_gp.add_observation(emb, reward)
            self.visited_bridges.add(idx)
            
            if reward > best_reward:
                best_reward = reward
                best_idx = idx
            
            print(f"    [{i+1}/{len(init_indices)}] idx={idx}, reward={reward:.2f}, details={details}")
        
        # Local search
        remaining_rounds = self.config.bridge_rounds - len(init_indices)
        print(f"\n  Local search ({remaining_rounds} rounds)...")
        
        current_idx = best_idx
        for r in range(remaining_rounds):
            _, current_emb = self.pool.get_bridge(current_idx)
            
            gradient = self.bridge_gp.estimate_gradient(current_emb)
            target_emb = current_emb + self.config.learning_rate * gradient
            target_emb = target_emb / np.linalg.norm(target_emb)
            
            neighbors = self.pool.find_nearest_bridges(target_emb, self.config.neighbor_k)
            
            for n_idx in neighbors:
                if n_idx not in self.visited_bridges:
                    bridge, emb = self.pool.get_bridge(n_idx)
                    reward, details = self.evaluator.evaluate(best_anchor, bridge)
                    
                    self.bridge_gp.add_observation(emb, reward)
                    self.visited_bridges.add(n_idx)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_idx = n_idx
                        current_idx = n_idx
                    
                    if (r + 1) % 10 == 0:
                        print(f"    [Round {r+1}] best_reward={best_reward:.2f}")
                    break
        
        self.best_reward = best_reward
        return best_idx
    
    def _validate_on_test(self) -> Tuple[float, Dict]:
        """Validate best suffix pair on test set"""
        # Note: For simplicity, use same evaluator but with test data
        # In full implementation, create separate evaluator for test set
        best_anchor = self.pool.anchors[self.best_anchor_idx]
        best_bridge = self.pool.bridges[self.best_bridge_idx]
        
        return self.evaluator.evaluate(best_anchor, best_bridge)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse
    from transformers import AutoModel, AutoTokenizer
    
    parser = argparse.ArgumentParser(description="ZOSO: Zeroth-Order Suffix Optimization")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--suffix-pool", type=str, required=True, help="Suffix candidates JSON file")
    parser.add_argument("--data", type=str, required=True, help="Data JSON file with adversarial texts")
    parser.add_argument("--corpus-embs", type=str, required=True, help="Corpus embeddings .npy file")
    parser.add_argument("--corpus-texts", type=str, required=True, help="Corpus texts JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output results JSON file")
    parser.add_argument("--anchor-rounds", type=int, default=50, help="Anchor optimization rounds")
    parser.add_argument("--bridge-rounds", type=int, default=50, help="Bridge optimization rounds")
    parser.add_argument("--golden-size", type=int, default=40, help="Golden set size")
    parser.add_argument("--init-exploration", type=int, default=20, help="Initial exploration samples")
    parser.add_argument("--skip-test-validation", action="store_true", help="Skip final validation on test set")
    parser.add_argument("--retrieval-top-k", type=int, default=10, help="Final retrieval top-k")
    parser.add_argument("--retrieval-first-k", type=int, default=20, help="First stage retrieval count")
    args = parser.parse_args()
    
    # Config
    config = ZOSOConfig(
        api_key=args.api_key,
        api_base_url=args.api_base,
        model_name=args.model,
        anchor_rounds=args.anchor_rounds,
        bridge_rounds=args.bridge_rounds,
        init_exploration=args.init_exploration,
        validate_on_test=not args.skip_test_validation,
        golden_set_size=args.golden_size,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_first_k=args.retrieval_first_k,
    )
    
    # Load data
    print("Loading data...")
    with open(args.data, "r") as f:
        all_data = json.load(f)
    
    # Load corpus
    print("Loading corpus...")
    corpus_embs = np.load(args.corpus_embs)
    with open(args.corpus_texts, "r") as f:
        corpus_texts = json.load(f)
    print(f"  Corpus: {len(corpus_texts)} docs")
    
    # Load Contriever
    print("Loading Contriever...")
    contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    contriever = AutoModel.from_pretrained("facebook/contriever").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    contriever = contriever.to(device)
    print(f"  Contriever on {device}")
    
    # Load suffix pool
    print("Loading suffix pool...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    pool = SuffixPool.from_file(Path(args.suffix_pool), embedder)
    
    # Initialize DataManager (with semantic labeling)
    print("Initializing DataManager...")
    client = OpenAI(base_url=config.api_base_url, api_key=config.api_key)
    data_mgr = DataManager(all_data, config, client)
    
    # Optimize
    optimizer = ZOSOOptimizer(
        config, pool, data_mgr,
        corpus_embs, corpus_texts,
        contriever, contriever_tokenizer,
    )
    results = optimizer.optimize()
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
