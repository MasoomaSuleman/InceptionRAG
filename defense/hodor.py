"""
HODOR (Heterogeneous Orthogonal Documents to Obstruct Reasoning) Defense

A defense mechanism against multi-document poisoning attacks like InceptionRAG.
By isolating each document and performing independent reasoning, HODOR breaks
the adversarial logic chain that relies on cross-document dependencies.

Core Idea:
- InceptionRAG exploits multi-document logical dependencies
- HODOR isolates documents to break the adversarial reasoning chain
- Each document generates an independent answer
- Final answer is determined by majority voting
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter


class HODORDefense:
    """
    HODOR Defense Implementation
    
    Document-Level Majority Voting to counter multi-document attacks.
    """
    
    def __init__(
        self,
        llm_client,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: int = 100,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize HODOR Defense.
        
        Args:
            llm_client: LLM client (OpenAI-compatible API)
            model: Model name for voting
            temperature: Generation temperature
            max_tokens: Max tokens per answer
            system_prompt: Custom system prompt (optional)
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return (
            "You are a helpful assistant. Answer the question based on the given context. "
            "Be concise and accurate. If the context doesn't contain relevant information, "
            "say 'I don't know'."
        )
    
    def apply(
        self,
        question: str,
        documents: List[str],
        verbose: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Apply HODOR defense to a query with retrieved documents.
        
        Args:
            question: User query
            documents: List of retrieved documents
            verbose: Print detailed logs
            
        Returns:
            (final_answer, defense_info)
        """
        defense_info = {
            "defense_type": "hodor",
            "num_documents": len(documents),
            "individual_answers": [],
            "vote_counts": {},
            "final_answer": "",
        }
        
        if not documents:
            return "I don't have enough information to answer.", defense_info
        
        individual_answers = []
        
        if verbose:
            print(f"[HODOR] Processing {len(documents)} documents individually...")
        
        for i, doc in enumerate(documents):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"CONTEXT:\n{doc}\n\nQUESTION: {question}"}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = (response.choices[0].message.content or "").strip()
                individual_answers.append(answer)
                
                if verbose:
                    preview = answer[:50].replace('\n', ' ')
                    print(f"  Doc {i}: {preview}...")
                    
            except Exception as e:
                if verbose:
                    print(f"  Doc {i}: Error - {e}")
                individual_answers.append("")
        
        defense_info["individual_answers"] = individual_answers
        
        # Normalize answers
        normalized = [a.lower().strip() for a in individual_answers if a.strip()]
        
        if not normalized:
            return "I don't have enough information to answer.", defense_info
        
        # Majority voting
        vote_counter = Counter(normalized)
        defense_info["vote_counts"] = dict(vote_counter.most_common())
        
        final_answer, max_votes = vote_counter.most_common(1)[0]
        defense_info["final_answer"] = final_answer
        defense_info["max_votes"] = max_votes
        defense_info["total_votes"] = len(normalized)
        
        if verbose:
            print(f"[HODOR] Final answer (votes={max_votes}/{len(normalized)}): {final_answer[:50]}...")
        
        return final_answer, defense_info
    
    def check_bypass(
        self,
        final_answer: str,
        target_answer: str,
        mode: str = "substring",
    ) -> bool:
        """
        Check if an attack bypassed the defense.
        
        Args:
            final_answer: The final answer from HODOR
            target_answer: The attacker's target answer
            mode: Check mode ("substring" or "exact")
            
        Returns:
            True if attack bypassed (target found in answer)
        """
        if mode == "substring":
            return target_answer.lower() in final_answer.lower()
        else:
            return target_answer.lower().strip() == final_answer.lower().strip()
