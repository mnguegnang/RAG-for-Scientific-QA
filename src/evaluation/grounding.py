"""
Grounding checks shared by context recall, faithfulness and ALCE citation
metrics: "is this sentence supported by these passages?".

GroundingJudge holds the metric logic (sentence splitting, citation-tag
stripping, aggregation); subclasses implement one primitive, ``_supported``.

  - PrometheusJudge (evaluate_rag.py) — True/False prompts to Prometheus 2.
  - MiniCheckJudge (here)             — MiniCheck-Flan-T5-Large, a 770M model
                                        trained specifically for grounding
                                        checks (Tang, Laban & Durrett, EMNLP
                                        2024, arXiv:2404.10774): GPT-4-level
                                        balanced accuracy on LLM-AggreFact
                                        (Table 2), above the T5-11B TRUE model
                                        ALCE uses (74.7 vs 61.0).

Why the switch (key_metrics_improvements.md, 2026-10-03, P1/D5): Prometheus'
True/False decisions agreed with an independent second judge at only
kappa=0.229 (ALCE entailment) and 0.292 (faithfulness). Prometheus 2 was
trained for rubric grading, not binary grounding.
"""
import logging
import re
from typing import List, Optional, Tuple

import nltk

_CITATION_TAG_RE = re.compile(r"\[Doc \d+(?:,\s*Doc \d+)*\]")


class GroundingJudge:
    """Metric logic on top of a single support primitive."""

    name = "base"

    def _supported(self, claim: str, passages: List[str], kind: str) -> Tuple[bool, Optional[str]]:
        """
        True if *claim* is supported by at least one of *passages* (or their
        combination). *kind* is "recall", "faithfulness" or "nli" (lets a
        prompt-based judge keep metric-specific wording). Returns
        (decision, evidence) where evidence is the prompt or a score string.
        """
        raise NotImplementedError

    # ── Context recall (RAGAS §3, sentence-level attribution) ───────────────

    def _score_context_recall_items(self, question: str, contexts: List[str],
                                    ground_truth: str) -> List[Tuple[str, bool]]:
        """[(gt_sentence, supported), ...]; [] when nothing can be scored."""
        if not contexts or not ground_truth:
            return []
        gt_sentences = [s for s in nltk.sent_tokenize(ground_truth) if len(s) >= 10]
        return [(s, self._supported(s, contexts, "recall")[0]) for s in gt_sentences]

    def score_context_recall(self, question: str, contexts: List[str],
                             ground_truth: str) -> float:
        """
        Fraction of ground-truth sentences supported by the retrieved passages.
        Es et al. (2023) RAGAS §3: recall = |{s ∈ GT : ∃p ∈ C, p supports s}| / |GT|.
        Every context passed to the generator is considered.
        """
        items = self._score_context_recall_items(question, contexts, ground_truth)
        return sum(r for _, r in items) / len(items) if items else float("nan")

    # ── Faithfulness (RAGAS §3, per-claim grounding) ─────────────────────────

    def _score_faithfulness_items(self, question: str, answer: str,
                                  contexts: List[str]) -> List[Tuple[str, bool]]:
        """
        [(claim, supported), ...] — one entry per answer sentence, citation
        tags stripped. A sentence that is empty after stripping counts as
        unsupported without a model call (nothing left to check).
        """
        if not contexts or not answer:
            return []
        items = []
        for sent in (s for s in nltk.sent_tokenize(answer) if len(s) >= 15):
            clean = _CITATION_TAG_RE.sub("", sent).strip()
            if not clean:
                items.append((sent, False))
                continue
            items.append((clean, self._supported(clean, contexts, "faithfulness")[0]))
        return items

    def score_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Fraction of answer claims supported by the retrieved context.
        Es et al. (2023) RAGAS §3: faithfulness is judged against the context
        the answer was generated from — i.e. *all* contexts the generator saw,
        not a prefix of them (the former contexts[:5] cut marked every correct
        citation of Doc 6-10 as unsupported).
        """
        items = self._score_faithfulness_items(question, answer, contexts)
        return sum(r for _, r in items) / len(items) if items else float("nan")

    # ── ALCE entailment (Gao et al. 2023 §3.3) ───────────────────────────────

    def check_nli_entailment(self, claim: str, cited_text: str) -> Tuple[bool, Optional[str]]:
        """
        Does *cited_text* (the concatenation of a sentence's cited passages)
        entail *claim*? Returns (result, evidence); evidence is None on the
        short-circuit paths where no model call was made.
        """
        if not cited_text.strip():
            return False, None
        clean = _CITATION_TAG_RE.sub("", claim).strip()
        if not clean:
            return False, None
        return self._supported(clean, [cited_text], "nli")


def _sent_tokenize_with_newlines(text: str) -> List[str]:
    """MiniCheck's own splitter: sentences, with newlines kept as tokens."""
    out = []
    for block in text.split("\n"):
        out.extend(nltk.sent_tokenize(block))
        out.append("\n")
    return out[:-1]


def chunk_document(doc: str, chunk_words: int = 500) -> List[str]:
    """
    Split *doc* into ~chunk_words-word chunks on sentence boundaries — the
    exact scheme MiniCheck-Flan-T5 was evaluated with (minicheck/inference.py,
    chunk_size=500 for flan-t5-large). A claim's support is the max over chunks.
    """
    chunks, current, count = [], [], 0
    for sentence in _sent_tokenize_with_newlines(doc) or [""]:
        n = len(sentence.split())
        if current and count + n > chunk_words:
            chunks.append(" ".join(current))
            current, count = [sentence], n
        else:
            current.append(sentence)
            count += n
    if current:
        chunks.append(" ".join(current))
    chunks = [c.replace(" \n ", "\n").strip() for c in chunks]
    return [c for c in chunks if c] or [""]


class MiniCheckJudge(GroundingJudge):
    """
    MiniCheck-Flan-T5-Large grounding checker, run in-process.

    Re-implements the reference inference (github.com/Liyan06/MiniCheck,
    minicheck/inference.py) instead of installing the package, whose
    unpinned dependencies could disturb the SPECTER2 stack (CLAUDE.md):
      input  = "predict: " + doc_chunk + </s> + claim   (max 2048 tokens)
      output = softmax over the first decoder step's logits for token ids
               3 ("unsupported") and 209 ("supported"); max over chunks;
               supported when p > 0.5.
    """

    name = "minicheck"
    CHECKPOINT = "lytang/MiniCheck-Flan-T5-Large"
    MAX_INPUT_TOKENS = 2048
    CHUNK_WORDS = 500
    THRESHOLD = 0.5
    # Batches are capped by batch_size x padded_length^2, because T5's
    # self-attention scores grow with the square of the input length. One
    # 2048-token input costs ~4 "units" of 1024^2; the budget below allows
    # 2 such inputs, or 16 inputs of ~700 tokens (the usual chunk size).
    ATTENTION_BUDGET = 8 * 1024 * 1024

    def __init__(self, checkpoint: str = CHECKPOINT, batch_size: int = 16,
                 device: Optional[str] = None, dtype: Optional[str] = None):
        import os
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # MINICHECK_DTYPE: "bfloat16" (default) or "float32" (the reference
        # implementation's precision). Validated 2026-10-04: bfloat16 flips
        # 0/131 labelled decisions (max |dp| = 0.025), gives identical
        # faithfulness on all 150 evaluation rows, and peaks at 3.2 GiB
        # reserved vs 4.8 GiB allocated in float32 — the float32 run hit CUDA
        # OOM next to the Prometheus vLLM server.
        if dtype is None:
            dtype = os.environ.get("MINICHECK_DTYPE", "bfloat16" if self.device == "cuda" else "float32")
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        logging.info("[MiniCheck] Loading %s on %s (%s) ...", checkpoint, self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, torch_dtype=torch_dtype).to(self.device).eval()
        self.batch_size = batch_size
        self._memo = {}

    def _batches(self, lengths: List[int]):
        """Index batches in length order, within batch_size and ATTENTION_BUDGET."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batch, longest = [], 0
        for i in order:
            longest_if_added = max(longest, lengths[i])
            if batch and (len(batch) + 1 > self.batch_size or
                          (len(batch) + 1) * longest_if_added ** 2 > self.ATTENTION_BUDGET):
                yield batch
                batch, longest_if_added = [], lengths[i]
            batch.append(i)
            longest = longest_if_added
        if batch:
            yield batch

    def _support_probs(self, inputs: List[str]) -> List[float]:
        """p(supported) per input, halving the batch on CUDA out-of-memory."""
        torch = self._torch
        encoded = self.tokenizer(inputs, max_length=self.MAX_INPUT_TOKENS, truncation=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        probs = [0.0] * len(inputs)
        pending = list(self._batches(lengths))
        while pending:
            batch = pending.pop(0)
            features = self.tokenizer.pad(
                {"input_ids": [encoded["input_ids"][i] for i in batch],
                 "attention_mask": [encoded["attention_mask"][i] for i in batch]},
                return_tensors="pt").to(self.device)
            decoder_input_ids = torch.zeros((len(batch), 1), dtype=torch.long, device=self.device)
            try:
                with torch.no_grad():
                    logits = self.model(**features, decoder_input_ids=decoder_input_ids).logits.squeeze(1)
            except torch.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                torch.cuda.empty_cache()
                half = len(batch) // 2
                logging.warning("[MiniCheck] CUDA out of memory on a batch of %d; retrying as %d + %d.",
                                len(batch), half, len(batch) - half)
                pending[:0] = [batch[:half], batch[half:]]
                continue
            batch_probs = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1].tolist()
            for i, p in zip(batch, batch_probs):
                probs[i] = p
        return probs

    def support_probability(self, doc: str, claim: str) -> float:
        key = (doc, claim)
        if key not in self._memo:
            eos = self.tokenizer.eos_token
            inputs = [f"predict: {chunk}{eos}{claim}"
                      for chunk in chunk_document(doc, self.CHUNK_WORDS)]
            self._memo[key] = max(self._support_probs(inputs))
        return self._memo[key]

    def _supported(self, claim: str, passages: List[str], kind: str) -> Tuple[bool, Optional[str]]:
        prob = self.support_probability("\n\n".join(passages), claim)
        return prob > self.THRESHOLD, f"minicheck p_support={prob:.3f}"
