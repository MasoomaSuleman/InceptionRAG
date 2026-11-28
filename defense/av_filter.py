import re
import yaml
import string
import numpy as np
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

ScoreFn = Callable[[str, Sequence[str]], List[float]]

@dataclass
class AVFilterConfig:
    use_avfilter: bool = False

    top_tokens: int = 1000000
    max_new_tokens: int = 24
    model: Any = None
    tokenizer: AutoTokenizer = None

    max_removals: int = 2
    sort_passes: int = 2
    keep_order: bool = True
    score_fn: ScoreFn = None
    threshold: float = 26.20711906167185

    @classmethod
    def from_yaml(cls, path: str, llm) -> "AVFilterConfig":
        """Load config from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        defense_cfg = data.get("defenses", {})
        avfilter_cfg = defense_cfg.get("avfilter", {})
        
        
        return cls(
            use_avfilter=avfilter_cfg.get("use_avfilter", False),
            tokenizer=llm._tokenizer,
            model=llm._model,
            top_tokens=avfilter_cfg.get("top_tokens", 1000000),
            max_new_tokens=avfilter_cfg.get("max_new_tokens", 24),
            max_removals=avfilter_cfg.get("max_removals", 2),
            sort_passes=avfilter_cfg.get("sort_passes", 2),
            keep_order=avfilter_cfg.get("keep_order", True),
        )


def init_score_fn(config: AVFilterConfig):
    if config.tokenizer is None or config.model is None:
        raise ValueError("AVFilter: tokenizer and model must be set in AVFilterConfig")

    
    config.score_fn = HFAttentionScorer(
            config.model, 
            config.tokenizer, 
            config.top_tokens, 
            config.max_new_tokens
        )
    
    return config


def calibrate_threshold(
        score_fn: ScoreFn, 
        benign_batches: Sequence[Tuple[str, Sequence[str]]], 
        target_fpr: float = 0.05,
        sort_passes: int = 2
    ) -> float:
    filter = AVFilter(
        score_fn,
        threshold=float("inf"),
        max_removals=0,
        sort_passes=sort_passes
    )

    vars_ = []
    for q, ps in benign_batches:
        _, _, meta = filter.filter(q, ps)
        v = meta["variances"]
        vars_.append(v[-1] if v else 0.0)

    threshold = float(np.quantile(vars_, 1.0 - target_fpr))
    return threshold


class AVFilter:
    def __init__(
            self,
            score_fn: ScoreFn,
            threshold: float,
            max_removals: int = 2,
            sort_passes: int = 2,
            keep_order: bool = True
    ):
        self.score_fn = score_fn
        self.threshold = threshold
        self.max_removals = max_removals
        self.sort_passes = sort_passes
        self.keep_order = keep_order

    def _get_text(self, document: dict) -> str:
        return str(document.get("text", "")).strip()

    def _score_norm(self, question, passages):
        scores = self.score_fn(question, passages)

        total = float(sum(scores))
        if total <= 0:
            return [100.0 / len(scores)] * len(scores)
        return [(100.0 * s) / total for s in scores]


    def filter(self, question: str, documents: Sequence[dict]) -> Tuple[List[dict], List[dict], Dict[str, Any]]:

        if not documents:
            return (
                list(documents), 
                [],
                {
                    "variances": [],
                    "final_scores": [],
                    "n_forward": 0
                }
            )

        valid_indices = []
        valid_texts = []
        for i, doc in enumerate(documents):
            text = self._get_text(doc)
            if text:
                valid_indices.append(i)
                valid_texts.append(text)

        n = len(valid_texts)
        if n <= 1:
            return (
                list(documents), 
                [],
                {
                    "variances": [],
                    "final_scores": [],
                    "n_forward": 0
                }
            )

        n_forward = 0
        og_texts: List[Tuple[int, str]] = list(enumerate(valid_texts))

        for _ in range(self.sort_passes):
            s = self._score_norm(question, [t for _, t in og_texts])
            n_forward += 1
            order = sorted(range(len(og_texts)), key=lambda i: s[i])
            og_texts = [og_texts[i] for i in order]

        removed_idxs: List[int] = []
        variances: list[float] = []
        scores: List[float] = []
        while True:
            scores = self._score_norm(question, [t for _, t in og_texts])
            n_forward += 1
            var = float(np.var(scores))
            variances.append(var)
            if var <= self.threshold or len(removed_idxs) >= self.max_removals or len(og_texts) <= 1:
                break

            j = int(np.argmax(scores))
            removed_idxs.append(og_texts[j][0])
            og_texts.pop(j)

        paired = list(zip(og_texts, scores))
        if self.keep_order:
            paired.sort(key=lambda x: x[0][0])

        kept_pairs = [p for p, _ in paired]
        reordered_scores = [s for _, s in paired]

        retained_valid_idxs = [valid_indices[i] for i, _ in kept_pairs]
        retained_docuemnts = [documents[i] for i in retained_valid_idxs]

        removed_valid_idxs = [valid_indices[i] for i in removed_idxs]
        removed_documents = [documents[i] for i in removed_valid_idxs]

        # reordered_scores = [scores[i] for i, _ in kept_pairs]

        metadata = {
            "variances": variances,
            "final_scores": reordered_scores,
            "n_forward": n_forward
        }

        return retained_docuemnts, removed_documents, metadata



class HFAttentionScorer:
    SYSTEM = "Answer the question using only the context. Give a short, direct answer."

    def __init__(
            self,
            model,
            tokenizer,
            top_tokens: Optional[int] = None,
            max_new_tokens: int = 24,
            answer_marker: Optional[str] = "",
            system_msg: Optional[str] = None,
            passage_header: str = "[{i}]",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_tokens = top_tokens
        self.max_new_tokens = max_new_tokens
        self.answer_marker = answer_marker.lower() if answer_marker else None
        self.system_msg = system_msg or self.SYSTEM
        self.passage_header = passage_header

        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("AVFilter: A *fast* tokenizer if required (offset mapping).")

    @staticmethod
    def _is_punct(s:str) -> bool:
        s = s.strip()
        return (not s) or all(c in set(string.punctuation) for c in s)

    def _step_attn(self, out, step, p):
        """Mean attention (over layers & heads) from generation step `step` onto the first p prompt tokens."""
        if step >= len(out.attentions):
            return None
        layers = out.attentions[step]
        if layers is None or len(layers) == 0:
            return None
        mats = [l[0, :, -1, :p] for l in layers if l is not None]
        if not mats:
            return None
        return torch.stack(mats).float().mean(dim=(0, 1)).cpu()

    def _fallback(self, out, gen_ids, p, in_mask, offsets, char_spans):
        total = torch.zeros(p)
        for i in range(0, len(gen_ids)-1):
            piece = self.tokenizer.decode([gen_ids[i]])
            if self._is_punct(piece):
                continue

            layers = out.attentions[i+1]

            attn = self._step_attn(out, i+1, p)
            if attn is None:
                continue
            total += attn

        return self._per_passage((total * in_mask).tolist(), offsets, char_spans)

    def _per_passage(self, token_attn, offsets, char_spans):
        scores = []
        for (cs, ce) in char_spans:
            vals = [token_attn[t] for t, (a, b) in enumerate(offsets) if b > cs and a < ce and b > a]
            if self.top_tokens is not None:
                vals = sorted(vals, reverse=True)[:self.top_tokens]
            scores.append(float(sum(vals)) if vals else 0.0)

        return scores
    

    def _build_prompt(self, question: str, passages: Sequence[str]):
        head = "CONTEXT:\n"
        body, spans, pos = "", [], len(head)
        separate = "\n\n"
        for i, p in enumerate(passages):
            hdr = self.passage_header.format(i=i+1)
            body += hdr
            pos += len(hdr)

            spans.append((pos, pos+len(p)))
            body += (p + separate)
            pos += len(p + separate)

        user_msg = f"{head}{body} QUESTION: {question}"
        msgs = [{"role": "system", "content": self.system_msg},
                {"role": "user", "content": user_msg}]

        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        off = prompt.index(user_msg)
        spans = [(s + off, e + off) for s, e in spans]
        if self.answer_marker:
            prompt += self.answer_marker
        return prompt, spans

    def prev__call__(self, question: str, passages: Sequence[str]) -> List[float]:
        prompt, char_spans = self._build_prompt(question, passages)

        encoded = self.tokenizer(
            prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
            )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        p = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=False,
                pad_token_id= self.tokenizer.eos_token_id
            )

        if not out.attentions or out.attentions[0] is None:
            raise RuntimeError("AVFilter: Model returned no attentions. Load it with attn_implementation='eager'.")

        gen_ids = out.sequences[0, p:].tolist()
        # input tokens that are pure punctuation contribute 0
        in_mask = torch.tensor(
            [0.0 if self._is_punct(self.tokenizer.decode([t])) else 1.0 
             for t in encoded["input_ids"][0].tolist()])

        total = torch.zeros(p)
        text_so_far, started, scored = "", self.answer_marker is None, 0
        for i in range(len(gen_ids)-1):
            piece = self.tokenizer.decode([gen_ids[i]])
            text_so_far += piece

            if gen_ids[i] == self.tokenizer.eos_token_id:
                break
            if "\n" in piece and scored > 0:
                break
            if not started:
                if self.answer_marker in text_so_far.lower():
                    started = True
                continue
            if self._is_punct(piece):
                continue

            # attentions: tuple over layers of [B, H, 1, kv]
            layers = out.attentions[i+1]
            attn = self._step_attn(out, i+1, p)
            if attn is None:
                continue
            total += attn
            scored += 1

        if scored == 0 and self.answer_marker is not None:
            print("fallback:", repr(self.tokenizer.decode(gen_ids)))
            return self._fallback(out, gen_ids, p, in_mask, offsets, char_spans)

        total = total * in_mask
        return self._per_passage(total.tolist(), offsets, char_spans)


    def __call__(self, question: str, passages: Sequence[str]) -> List[float]:
            prompt, char_spans = self._build_prompt(question, passages)
    
            encoded = self.tokenizer(
                prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
                )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            ids = encoded["input_ids"]
            p = ids.shape[1]
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tokenizer.eos_token_id}
            eos_ids.discard(None)
            dev = ids.device
    
            with torch.inference_mode():
                # 1. prefill: no attentions kept
                out = self.model(
                    **encoded,
                    use_cache=True,
                    output_attentions=False
                )
                past = out.past_key_values
                tok = out.logits[:, -1].argmax(-1, keepdim=True)
                del out

                total = torch.zeros(p, device=dev)
                fb_total = torch.zeros(p, device=dev)
                text_so_far, started, scored = "", self.answer_marker is None, 0

                # 2. decode with cache
                for _ in range(self.max_new_tokens - 1):
                    t = tok.item()
                    if t == eos_ids:
                        break

                    piece = self.tokenizer.decode([t])
                    text_so_far += piece

                    if "\n" in piece and scored > 0:
                        break

                    if not started and self.answer_marker in text_so_far.lower():
                        started = True
                        need_attn, count = False, False
                    else:
                        need_attn = not self._is_punct(piece)
                        count = started and need_attn

                    out = self.model(
                        input_ids=tok,
                        past_key_values=past,
                        use_cache=True,
                        output_attentions=need_attn
                    )
                    past = out.past_key_values

                    if need_attn and not out.attentions:
                        raise RuntimeError("AVFilter: Model returned no attentions. Load it with attn_implementation='eager'.")

                    if need_attn and out.attentions:
                        a = torch.stack([l[0, :, -1, :p] for l in out.attentions if l is not None])
                        a = a.float().mean(dim=(0,1))
                        fb_total += a
                        if count:
                            total += a
                            scored += 1

                    tok = out.logits[:, -1].argmax(-1, keepdims=True)
                    del out

                del past

            if scored == 0 and self.answer_marker is not None:
                total = fb_total

            in_mask = torch.tensor(
                [0.0 if self._is_punct(self.tokenizer.decode([t])) else 1.0 for t in ids[0].tolist()]
            )
            total = total.cpu() * in_mask
            return self._per_passage(total.tolist(), offsets, char_spans)