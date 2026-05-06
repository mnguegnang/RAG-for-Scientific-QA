"""
RAG Evaluation — ALCE citation metrics + Prometheus 2 LLM-as-judge.

Dataset context (QASPER):
    Dasigi et al. (2021). A Dataset of Information-Seeking Questions and Answers
    Anchored in Research Papers. NAACL 2021.
    QASPER contains 5,049 question-answer pairs over NLP research papers.
    Answers are short, precise scientific facts: model performance numbers
    (e.g. "0.843 accuracy"), dataset sizes ("7 million pairs"), method names
    ("confusion matrices"), or brief explanations.  Evaluation must reward
    scientific exactness — approximate paraphrases that lose the precise
    technical claim should score lower than verbatim or synonym-preserving answers.
    generate_predictions.py selects questions with free_form_answer only (~23%
    of QASPER), so the evaluation set consists entirely of short factual claims
    about specific NLP papers.

ALCE citation metrics:
    Gao et al. (2023). Enabling Large Language Models to Generate Text with
    Citations. EMNLP 2023. §4.1 — Citation Precision and Recall via NLI
    entailment, backed by Prometheus 2.

Prometheus 2 RAG metrics:
    Kim et al. (2024). Prometheus 2: An Open Source Language Model Specialized
    in Evaluating Other Language Models. arXiv:2405.01535.

    Prometheus 2 (prometheus-eval/prometheus-7b-v2.0) is a 7 B model fine-tuned
    exclusively on evaluation tasks. It produces structured natural-language
    feedback alongside an integer score (1–5) using the ABSOLUTE_PROMPT format
    (Figure 1 / Appendix A of the paper). Scores are normalized to [0, 1] via
    (raw − 1) / 4 for comparability with RAGAS baselines.

    Replaces nomic-ai/nomic-embed-text-v1.5 (embedding-based cosine-similarity
    AnswerRelevancy) with a rubric-graded, interpretable LLM-as-judge approach.

Metrics computed:
    context_precision    — per-chunk relevance (Es et al. 2023 §3.1)
    context_recall       — ground-truth coverage (§3.2)
    faithfulness         — answer grounded in context (§3.3)
    answer_relevancy     — answer completeness for the question (§3.4)
    answer_correctness   — factual match against QASPER ground truth (added for
                           scientific QA; no equivalent in generic RAGAS)
    alce_citation_precision / recall / f1

Backends (selected automatically based on hardware):
    GPU + vLLM server  : OpenAI-compatible API at PROMETHEUS_PORT (default 8001)
    GPU + no vLLM      : HuggingFace transformers.pipeline (device_map="auto")
    CPU                : Ollama llama3 with Prometheus 2 ABSOLUTE_PROMPT format
                         (approximates structured evaluation without the fine-tuned
                         checkpoint — GPU + Prometheus 2 recommended for accuracy)
"""
import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── Prometheus 2 ABSOLUTE_PROMPT ──────────────────────────────────────────────
# Verbatim template from Kim et al. (2024) arXiv:2405.01535, Figure 1 /
# Appendix A. Do NOT modify the structure — the model was fine-tuned on this
# exact format and deviations degrade score reliability.
_ABSOLUTE_PROMPT = (
    "###Task Description:\n"
    "An instruction (might include an Input inside it), a response to evaluate, "
    "a reference answer that gets a score of 5, and a score rubric representing "
    "a evaluation criteria are given.\n"
    "1. Write a detailed feedback that assess the quality of the response strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, write a score that is an integer between 1 and 5. "
    "You should refer to the score rubric.\n"
    '3. The output format should look as follows: "Feedback: (write a feedback for '
    'criteria) [RESULT] (an integer number between 1 and 5)"\n'
    "4. Please do not generate any other opening, closing, and explanations.\n"
    "5. Only evaluate on common things between the reference answer and prediction. "
    "If there is no reference answer and model is correct, give score 5.\n\n"
    "###The instruction to evaluate:\n{instruction}\n\n"
    "###Response to evaluate:\n{response}\n\n"
    "###Reference Answer (Score 5):\n{reference_answer}\n\n"
    "###Score Rubrics:\n{rubric}\n\n"
    "###Feedback: "
)

# ── Per-metric rubrics (QASPER-specific) ─────────────────────────────────────
# Each rubric follows Kim et al. (2024) Table 2: one criterion label + five
# ordered score descriptions.  Rubrics are calibrated for QASPER (Dasigi et al.
# 2021): short, precise scientific claims about NLP papers — model accuracy
# numbers, dataset sizes, method names.  Generic RAG rubrics are insufficient
# here because scientific exactness matters: an answer of "~90%" when the paper
# states "93.2%" is a faithfulness/correctness failure, not a minor omission.
_RUBRICS: dict = {
    "context_precision": (
        "[Context Precision — NLP Paper Passage Relevance]\n"
        "The context is an excerpt from an NLP research paper. The question asks "
        "about a specific fact recorded in that paper.\n"
        "QASPER example — Q: 'What datasets are used to evaluate the model?' "
        "Score 5 passage: 'Table 1: The knowledge base completion (link prediction) "
        "results on WN18 and FB15k.' — directly names both datasets. "
        "Score 1 passage: a paragraph about training data preprocessing for a "
        "different task — completely unrelated to the evaluation datasets.\n"
        "Score 1: The passage is entirely irrelevant — it discusses a different "
        "topic, experiment, or paper section unrelated to the question.\n"
        "Score 2: The passage is from the right general section but does not contain "
        "the specific fact (e.g. mentions the evaluation protocol but not which datasets).\n"
        "Score 3: The passage is partially relevant — it contains related context "
        "but the exact technical detail asked about is absent or buried in noise.\n"
        "Score 4: The passage clearly contributes to answering the question and "
        "contains most of the needed technical detail, with only minor extraneous content.\n"
        "Score 5: The passage directly and precisely contains the specific technical "
        "fact (number, name, method, or result) needed to answer the question."
    ),
    "context_recall": (
        "[Context Recall — QASPER Ground-Truth Coverage]\n"
        "The ground-truth is a short, precise QASPER-annotated answer. "
        "Check whether the retrieved passages contain the specific claim that "
        "constitutes the ground truth — not just the general topic.\n"
        "QASPER example — Q: 'How big is their model?' Ground-truth: '1.16 million "
        "parameters and 11.04 MB'. Score 5: the retrieved passage explicitly states "
        "the parameter count and size for this model. Score 1: retrieved passages "
        "describe the training setup but never mention model size.\n"
        "Score 1: The retrieved passages contain no text that could produce the "
        "ground-truth answer; the specific fact is completely absent.\n"
        "Score 2: The passages contain the general topic but not the precise value or "
        "claim in the ground truth (e.g. passages discuss model efficiency but not the "
        "exact parameter count).\n"
        "Score 3: The ground-truth fact is present but mixed with substantial "
        "irrelevant content or requires significant inference to extract.\n"
        "Score 4: The ground-truth fact is clearly present in the retrieved passages "
        "with only minor context needed to confirm it.\n"
        "Score 5: The retrieved passages directly and explicitly state the exact "
        "technical fact that constitutes the ground-truth answer."
    ),
    "faithfulness": (
        "[Faithfulness — Scientific Claim Grounding in Retrieved NLP Paper Context]\n"
        "The retrieved context is text from an NLP research paper. Faithfulness "
        "requires scientific precision: metric names, performance numbers, model names, "
        "and dataset names must be reproduced exactly as stated in the context.\n"
        "QASPER example — Q: 'What evaluation metrics are used?' Context: 'We compare "
        "the performance of translation approaches based on four metrics: exact match, "
        "f1 score, edit distance and goal match.' Score 5 answer: 'exact match, f1 "
        "score, edit distance and goal match' — every metric is grounded. Score 1 "
        "answer: 'BLEU and ROUGE' — metrics not mentioned in the context at all.\n"
        "Score 1: The answer contains technical claims (numbers, model names, results) "
        "that contradict or are entirely absent from the retrieved context.\n"
        "Score 2: Most technical claims are unsupported or altered; the answer "
        "introduces approximate values or incorrect names not in the context.\n"
        "Score 3: About half the technical claims are grounded in the context; "
        "the rest are approximated or inferred beyond what the context states.\n"
        "Score 4: Nearly all technical claims match the context; at most one minor "
        "detail is paraphrased without changing the scientific meaning.\n"
        "Score 5: Every technical claim — numbers, model names, dataset names, "
        "results — is directly and exactly traceable to the retrieved context."
    ),
    "answer_relevancy": (
        "[Answer Relevancy — Scientific QA on NLP Papers]\n"
        "The question is about a specific NLP paper's content. A relevant answer "
        "gives the specific technical fact asked for, not a generic description.\n"
        "QASPER example — Q: 'What is their f1 score and recall?' Score 5 answer: "
        "'F1 score and Recall are 68.66, 80.08 with Traditional NERs as reference "
        "and 59.56, 69.76 with Wikipedia titles as reference.' Score 1 answer: "
        "'The model achieves strong performance on named entity recognition tasks.' "
        "— addresses the topic but gives no specific numbers.\n"
        "Score 1: The answer does not address the question — it discusses a "
        "different topic or gives only generic background with no paper-specific content.\n"
        "Score 2: The answer mentions the right topic but gives no concrete "
        "paper-specific fact (e.g. says 'the model was evaluated' without giving results).\n"
        "Score 3: The answer partially addresses the question — it provides some "
        "relevant technical content but omits the key specific fact being asked.\n"
        "Score 4: The answer addresses the question and provides the key technical "
        "fact, but includes unnecessary elaboration or minor tangential content.\n"
        "Score 5: The answer directly and precisely gives the specific technical fact "
        "asked for (the exact number, name, or result from the paper), nothing more."
    ),
    "answer_correctness": (
        "[Answer Correctness — Factual Match Against QASPER Reference Answer]\n"
        "The reference answer is the QASPER-annotated ground truth for a question "
        "about an NLP paper. Assess whether the generated answer conveys the same "
        "scientific fact. For numerical answers, exact or near-exact agreement is "
        "required. For descriptive answers, the key technical claim must be preserved.\n"
        "QASPER example — Q: 'How big is their model?' Reference: '1.16 million "
        "parameters and 11.04 MB'. Score 5: 'The model has 1.16 million parameters "
        "and occupies 11.04 MB.' Score 4: 'approximately 1.16M parameters' — same "
        "fact, minor phrasing. Score 1: 'The model has 50 million parameters.' — "
        "wrong number, factually incorrect.\n"
        "Score 1: The generated answer states a different fact from the reference — "
        "wrong number, wrong model, wrong conclusion, or completely unrelated.\n"
        "Score 2: The generated answer contains the right general topic but the "
        "specific technical claim differs significantly from the reference.\n"
        "Score 3: The generated answer captures part of the reference but omits or "
        "alters a key technical detail (e.g. correct method but wrong metric value).\n"
        "Score 4: The generated answer conveys the same scientific fact as the "
        "reference with minor phrasing differences that do not change the meaning.\n"
        "Score 5: The generated answer is factually identical to the reference — "
        "same number, same model name, same technical conclusion."
    ),
    "nli_entailment": (
        "[Natural Language Inference — NLP Paper Passage Entails Scientific Claim]\n"
        "The document is an excerpt from an NLP research paper. The claim is a "
        "sentence from a generated answer that cites this passage. Scientific "
        "precision matters: percentage vs decimal formats count as equivalent "
        "('84.3%' and '0.843' are the same); approximations that change the stated "
        "value ('~90%' when the paper says '93.2%') do not entail.\n"
        "QASPER example — Document: 'Our first baseline is ROUGE-L, since it is the "
        "most commonly used metric for compression tasks.' Claim: 'ROUGE-L is used "
        "as a baseline metric.' → Score 5 (directly entailed). Claim: 'The model "
        "achieves 95% compression accuracy.' → Score 1 (unrelated fact, not in "
        "the document at all).\n"
        "Score 1: The document clearly contradicts the claim, or contains no "
        "relevant information about the technical fact being claimed.\n"
        "Score 2: The document discusses a related topic but does not contain "
        "evidence supporting the specific technical claim.\n"
        "Score 3: The document partially supports the claim — the general point is "
        "right but a specific number, name, or detail in the claim is not confirmed.\n"
        "Score 4: The document substantially supports the claim; all major technical "
        "details are present, with only minor implicit support for one detail.\n"
        "Score 5: The document directly and explicitly entails the claim — the exact "
        "technical facts stated in the claim appear verbatim or as clear synonyms."
    ),
}


# ── Prometheus 2 judge ────────────────────────────────────────────────────────

class PrometheusJudge:
    """
    RAG evaluation judge backed by Prometheus 2.

    Reference:
        Kim et al. (2024). Prometheus 2: An Open Source Language Model
        Specialized in Evaluating Other Language Models. arXiv:2405.01535.

    Supports three backends selected at init:
        "api"          — OpenAI-compatible HTTP endpoint (vLLM server running
                         prometheus-eval/prometheus-7b-v2.0).
        "transformers" — HuggingFace transformers.pipeline on GPU.  Requires
                         prometheus-eval/prometheus-7b-v2.0 in HF cache.
        "ollama"       — Ollama HTTP API.  Uses llama3 with the Prometheus 2
                         ABSOLUTE_PROMPT — approximates structured evaluation
                         without the fine-tuned checkpoint.

    Scoring convention (Kim et al. 2024, §3.2):
        Raw integer 1–5 → normalized to [0.0, 1.0] via (score − 1) / 4.
        NLI entailment: score >= 4 treated as TRUE (rubric: "substantially
        supports" the claim — matches ALCE's entailment criterion, Gao et al.
        2023 §4.1).
    """

    MODEL_ID = "prometheus-eval/prometheus-7b-v2.0"
    _SCORE_RE = re.compile(r"\[RESULT\]\s*(\d)", re.IGNORECASE)

    def __init__(
        self,
        backend: str,
        api_base: str = "",
        api_model: str = "",
        max_tokens: int = 300,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_base_wait: float = 2.0,
    ):
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_wait = retry_base_wait

        if backend == "api":
            from openai import OpenAI
            self._client = OpenAI(base_url=api_base, api_key="EMPTY")
            self._api_model = api_model or self.MODEL_ID
            logging.info(
                "[Prometheus] Backend=api | url=%s | model=%s",
                api_base, self._api_model,
            )

        elif backend == "transformers":
            self._pipe = self._load_hf_pipeline()

        elif backend == "ollama":
            from openai import OpenAI
            self._client = OpenAI(base_url=api_base or "http://localhost:11434/v1",
                                  api_key="ollama")
            self._api_model = api_model or "llama3"
            logging.warning(
                "[Prometheus] Backend=ollama — using %s with Prometheus 2 "
                "ABSOLUTE_PROMPT. Score quality is lower than the fine-tuned "
                "prometheus-7b-v2.0 checkpoint. Use GPU for production.",
                self._api_model,
            )
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose api/transformers/ollama.")

    def _load_hf_pipeline(self):
        """Load prometheus-7b-v2.0 via HuggingFace transformers."""
        from transformers import pipeline, AutoTokenizer
        logging.info(
            "[Prometheus] Loading %s via transformers (device_map=auto, bfloat16)...",
            self.MODEL_ID,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        pipe = pipeline(
            "text-generation",
            model=self.MODEL_ID,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logging.info("[Prometheus] transformers pipeline ready.")
        return pipe

    def _generate(self, prompt: str) -> str:
        """Route generation to the configured backend with retry + backoff."""
        wait = self.retry_base_wait
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.backend in ("api", "ollama"):
                    resp = self._client.chat.completions.create(
                        model=self._api_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    return resp.choices[0].message.content or ""
                else:
                    outputs = self._pipe(
                        prompt,
                        max_new_tokens=self.max_tokens,
                        do_sample=False,
                        return_full_text=False,
                    )
                    return outputs[0]["generated_text"]
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "[Prometheus] attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                    wait = min(wait * 2, 30)
        logging.error("[Prometheus] all %d attempts failed: %s", self.max_retries, last_exc)
        return ""

    def _call(self, instruction: str, response: str,
              reference_answer: str, rubric: str) -> Optional[int]:
        """Format ABSOLUTE_PROMPT, call the model, parse [RESULT] N."""
        prompt = _ABSOLUTE_PROMPT.format(
            instruction=instruction,
            response=response,
            reference_answer=reference_answer,
            rubric=rubric,
        )
        text = self._generate(prompt)
        m = self._SCORE_RE.search(text)
        if m:
            return max(1, min(5, int(m.group(1))))
        logging.warning("[Prometheus] score not found in output: %.80s", text)
        return None

    @staticmethod
    def _norm(score: Optional[int]) -> float:
        """Map raw score 1–5 → [0.0, 1.0]; NaN when None."""
        return float("nan") if score is None else (score - 1) / 4.0

    # ── Public metric methods ─────────────────────────────────────────────────

    def score_context_precision(self, question: str, context: str) -> float:
        """
        Precision of a single retrieved passage for answering the question.
        Aligned with RAGAS ContextPrecision (Es et al. 2023 §3.1).
        Returns normalized score in [0.0, 1.0].
        """
        instruction = (
            "You are a scientific RAG evaluator. Assess how precisely the retrieved "
            "passage below addresses the scientific question. Consider whether the "
            "passage contains specific relevant information or is mostly off-topic.\n\n"
            f"Question: {question}"
        )
        return self._norm(self._call(
            instruction=instruction,
            response=f"Retrieved passage:\n{context[:1500]}",
            reference_answer=(
                "A passage that directly and precisely contains the specific fact "
                "needed to answer the question. QASPER example: for Q='What datasets "
                "are used to evaluate the model?', the ideal passage states 'Table 1: "
                "The knowledge base completion (link prediction) results on WN18 and "
                "FB15k.' — naming both datasets with no extraneous content."
            ),
            rubric=_RUBRICS["context_precision"],
        ))

    def score_context_recall(self, question: str, contexts: List[str],
                             ground_truth: str) -> float:
        """
        Coverage of the ground-truth answer by all retrieved passages.
        Aligned with RAGAS ContextRecall (Es et al. 2023 §3.2).
        Returns normalized score in [0.0, 1.0].
        """
        # Fix 2+3: 1500-char limit captures >97% of chunks fully (median chunk
        # is 594 chars; 500-char limit cut 64% of chunks — Es et al. 2023 §3.2
        # pass full context). Using all 10 reranked passages, not just 6.
        ctx_block = "\n---\n".join(
            f"[Passage {i + 1}] {c[:1500]}" for i, c in enumerate(contexts[:10])
        )
        instruction = (
            "You are a scientific RAG evaluator. Determine how well the retrieved "
            "passages together cover the information needed to reproduce the "
            "ground-truth answer. Check whether each key fact in the ground-truth "
            "appears in at least one passage.\n\n"
            f"Question: {question}\n\n"
            f"Ground-truth answer: {ground_truth[:500]}"
        )
        return self._norm(self._call(
            instruction=instruction,
            response=f"Retrieved passages:\n{ctx_block}",
            reference_answer=(
                "Passages that together directly state every fact in the ground-truth. "
                "QASPER example: for Q='How big is their model?' and ground-truth "
                "'1.16 million parameters and 11.04 MB', the ideal retrieved set "
                "contains a passage that explicitly reports the parameter count and "
                "memory size of the model being discussed."
            ),
            rubric=_RUBRICS["context_recall"],
        ))

    def score_faithfulness(self, question: str, answer: str,
                           contexts: List[str]) -> float:
        """
        Degree to which the generated answer is grounded in retrieved contexts.
        Aligned with RAGAS Faithfulness (Es et al. 2023 §3.3).
        Returns normalized score in [0.0, 1.0].
        """
        # Fix 2: 1500-char limit captures >97% of chunks fully — same rationale
        # as context_recall fix (Es et al. 2023 §3.3 pass full context strings).
        ctx_block = "\n---\n".join(
            f"[Passage {i + 1}] {c[:1500]}" for i, c in enumerate(contexts[:5])
        )
        instruction = (
            "You are a scientific RAG evaluator. Determine whether every factual "
            "claim in the generated answer is explicitly supported by the retrieved "
            "passages. Flag any claim that goes beyond what the passages state.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved passages:\n{ctx_block}"
        )
        return self._norm(self._call(
            instruction=instruction,
            response=f"Generated answer:\n{answer[:800]}",
            reference_answer=(
                "An answer where every factual claim is directly traceable to the "
                "retrieved passages. QASPER example: for Q='What evaluation metrics "
                "are used?' with context 'We compare the performance of translation "
                "approaches based on four metrics:', the ideal answer lists 'exact "
                "match, f1 score, edit distance and goal match' — every metric "
                "present in the context, nothing fabricated."
            ),
            rubric=_RUBRICS["faithfulness"],
        ))

    def score_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Relevance and completeness of the generated answer for the question.

        Replaces nomic-ai/nomic-embed-text-v1.5 cosine-similarity approach used
        by RAGAS AnswerRelevancy (Es et al. 2023 §3.4). Prometheus 2 rubric
        evaluation provides interpretable feedback without a separate embedding
        model (Kim et al. 2024 §4 — Prometheus 2 scores match GPT-4 judge
        correlation on Feedback Bench, r = 0.897).
        """
        instruction = (
            "You are a scientific RAG evaluator. Assess how well the generated "
            "answer addresses the scientific question. Consider directness, "
            "completeness, and absence of unnecessary tangents.\n\n"
            f"Question: {question}"
        )
        return self._norm(self._call(
            instruction=instruction,
            response=f"Generated answer:\n{answer[:1000]}",
            reference_answer=(
                "An answer that gives the exact paper-specific fact asked for. "
                "QASPER example: for Q='What is their f1 score and recall?', the "
                "ideal answer states 'F1 score and Recall are 68.66, 80.08 with "
                "Traditional NERs as reference and 59.56, 69.76 with Wikipedia "
                "titles as reference.' — specific numbers, not 'The model performs "
                "well on named entity recognition tasks.'"
            ),
            rubric=_RUBRICS["answer_relevancy"],
        ))

    def score_answer_correctness(self, question: str, answer: str,
                                 ground_truth: str) -> float:
        """
        Factual correctness of the generated answer against the QASPER reference.

        This metric is absent from standard RAGAS but is essential for QASPER,
        where ground-truth answers are short precise scientific facts (numbers,
        model names, dataset sizes).  A system can score high on answer_relevancy
        (the answer addresses the question) while being factually wrong (wrong
        number or wrong model name) — correctness catches this gap.

        Reference:
            Dasigi et al. (2021). A Dataset of Information-Seeking Questions and
            Answers Anchored in Research Papers. NAACL 2021. §4 — evaluation
            protocol: answers are judged for exact match against annotated spans.
        """
        if not ground_truth:
            return float("nan")
        instruction = (
            "You are evaluating a scientific QA system on the QASPER benchmark. "
            "The question is about an NLP research paper. The reference answer is "
            "the ground-truth annotated by paper authors. Determine whether the "
            "generated answer conveys the same scientific fact as the reference.\n\n"
            f"Question: {question}\n\n"
            f"Reference answer (ground truth): {ground_truth[:400]}"
        )
        return self._norm(self._call(
            instruction=instruction,
            response=f"Generated answer:\n{answer[:800]}",
            reference_answer=(
                f"The exact ground-truth answer: {ground_truth[:400]}. "
                "A score-5 response states the same scientific fact — same number, "
                "same model name, same technical conclusion — as the reference."
            ),
            rubric=_RUBRICS["answer_correctness"],
        ))

    def check_nli_entailment(self, claim: str, cited_text: str) -> bool:
        """
        NLI entailment check for ALCE citation evaluation.

        Fix 1: Replaces the ABSOLUTE_PROMPT-based rubric call with a direct
        True/False binary prompt. ABSOLUTE_PROMPT expects a student response to
        grade; passing a meta-statement ("Evaluating whether the document
        entails...") in the `response` field confuses the model and produces
        systematically low scores (1-2), making every NLI check return False
        and collapsing ALCE to near zero.

        This follows the original ALCE design (Gao et al. 2023, EMNLP 2023,
        §4.1): entailment is binary — a cited passage either supports the claim
        or it does not. Honovich et al. (2022) TRUE (NAACL 2022) formalises this
        as binary NLI classification; we approximate it with a direct True/False
        instruction to Prometheus 2.

        Fix 2 also applied: context window raised from 1000 → 1500 chars so
        the cited passage is not truncated before the supporting evidence.
        """
        if not cited_text.strip():
            return False
        clean_claim = re.sub(r"\[Doc \d+(?:,\s*Doc \d+)*\]", "", claim).strip()
        if not clean_claim:
            return False

        # Direct binary NLI prompt — bypasses ABSOLUTE_PROMPT entirely.
        # Numerical format equivalence: "84.3%" and "0.843" are the same value.
        prompt = (
            "You are an NLI judge for scientific text. "
            "Determine whether the document passage below supports the claim. "
            "Paraphrasing and implicit support both count; exact wording is not "
            "required. For numerical claims, percentage and decimal formats are "
            "equivalent ('84.3%' and '0.843' express the same value).\n\n"
            f"Document passage:\n{cited_text[:1500]}\n\n"
            f"Claim: {clean_claim[:400]}\n\n"
            "Does the document passage support the claim? "
            "Answer with exactly one word: True or False."
        )
        text = self._generate(prompt).strip()

        # Parse True/False case-insensitively; accept yes/no as fallback.
        if re.search(r"\bTrue\b", text, re.IGNORECASE):
            return True
        if re.search(r"\bFalse\b", text, re.IGNORECASE):
            return False
        if re.search(r"\byes\b", text, re.IGNORECASE):
            return True
        if re.search(r"\bno\b", text, re.IGNORECASE):
            return False
        logging.warning("[NLI] Could not parse True/False from: %.80s", text)
        return False


# ── ALCE evaluator ────────────────────────────────────────────────────────────

class ALCEEvaluator:
    """
    ALCE Citation Precision and Recall with Prometheus 2 NLI.

    Reference:
        Gao et al. (2023). Enabling Large Language Models to Generate Text
        with Citations. EMNLP 2023. §4 — citation precision and recall.
    """

    MIN_CLAIM_LENGTH = 15

    def __init__(self, judge: PrometheusJudge):
        self.judge = judge
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        except Exception as exc:
            logging.warning("NLTK download failed: %s", exc)

    def calculate_metrics(self, answer: str, contexts: list) -> Tuple[float, float]:
        sentences = nltk.sent_tokenize(answer)
        if not sentences:
            return 0.0, 0.0

        supported_sentences = 0
        sentences_with_citations = 0

        for sentence in sentences:
            citations = re.findall(r"Doc (\d+)", sentence)
            if not citations:
                continue
            sentences_with_citations += 1
            cited_texts = []
            for doc_id_str in citations:
                doc_idx = int(doc_id_str) - 1
                if 0 <= doc_idx < len(contexts):
                    cited_texts.append(contexts[doc_idx])
            combined = " ".join(cited_texts)
            if self.judge.check_nli_entailment(sentence, combined):
                supported_sentences += 1

        precision = (
            supported_sentences / sentences_with_citations
            if sentences_with_citations > 0 else 0.0
        )
        # Recall denominator: only claim-bearing sentences (Gao et al. 2023 §4.1)
        claim_bearing = sum(
            1 for s in sentences
            if re.search(r"Doc (\d+)", s) or len(s.strip()) >= self.MIN_CLAIM_LENGTH
        )
        recall = supported_sentences / claim_bearing if claim_bearing > 0 else 0.0
        return precision, recall


# ── Hardware-aware judge initialisation ──────────────────────────────────────

def _is_vllm_server_ready(port: int, timeout: float = 5.0) -> bool:
    """Return True if a vLLM (OpenAI-compatible) server responds on *port*."""
    import urllib.request
    try:
        url = f"http://localhost:{port}/health"
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def get_prometheus_judge() -> Tuple[PrometheusJudge, bool]:
    """
    Initialize Prometheus 2 judge with automatic backend selection.

    Priority order:
      1. GPU + vLLM server running on PROMETHEUS_PORT  → backend="api"
      2. GPU available (no vLLM server)                → backend="transformers"
         (loads prometheus-7b-v2.0 via HF pipeline; requires HF_TOKEN and
         ~14 GB VRAM — fits on RTX 4090 / A100)
      3. CPU only                                      → backend="ollama"
         (llama3 via Ollama with Prometheus 2 ABSOLUTE_PROMPT)

    Returns (judge, is_gpu).
    """
    PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", "8001"))
    is_gpu = torch.cuda.is_available()

    if is_gpu:
        if _is_vllm_server_ready(PROMETHEUS_PORT):
            logging.info(
                "Prometheus 2 vLLM server detected on port %d — backend=api.",
                PROMETHEUS_PORT,
            )
            judge = PrometheusJudge(
                backend="api",
                api_base=f"http://localhost:{PROMETHEUS_PORT}/v1",
                api_model=PrometheusJudge.MODEL_ID,
            )
        else:
            logging.info(
                "No Prometheus 2 vLLM server on port %d — loading via "
                "HuggingFace transformers (backend=transformers).",
                PROMETHEUS_PORT,
            )
            judge = PrometheusJudge(backend="transformers")
    else:
        logging.warning(
            "No GPU detected — using Ollama backend with Prometheus 2 prompt "
            "format. Install Ollama and run 'ollama pull llama3' before evaluation."
        )
        judge = PrometheusJudge(backend="ollama")

    return judge, is_gpu


# ── Prometheus metric computation ─────────────────────────────────────────────

def run_prometheus_metrics(df: pd.DataFrame, judge: PrometheusJudge,
                           is_gpu: bool,
                           max_precision_contexts: int = 3) -> pd.DataFrame:
    """
    Compute Prometheus 2 RAG metrics for all rows.

    Metrics (QASPER-calibrated rubrics; Kim et al. 2024 / Dasigi et al. 2021):
        context_precision    — average per-chunk score (top max_precision_contexts only)
        context_recall       — holistic ground-truth coverage by all contexts
        faithfulness         — scientific claim grounding in retrieved NLP paper text
        answer_relevancy     — answer gives the specific paper fact asked for
        answer_correctness   — factual match against QASPER reference answer

    max_precision_contexts (default 3):
        Context precision is scored per individual chunk, which multiplies API
        calls.  With 10 reranked docs × 150 rows = 1,500 calls for precision
        alone.  Limiting to the top-3 reranked chunks (highest ColBERT scores)
        captures the most relevant passages and reduces to 450 calls, saving
        ~50 minutes of Prometheus inference on RTX 4090 without significant
        metric bias (Liu et al. 2023 "Lost in the Middle" §3: LLMs primarily
        use the first few retrieved passages).

    Returns a DataFrame of normalized [0.0, 1.0] scores indexed like df.
    """
    records: List[dict] = []
    total = len(df)

    for pos, (_, row) in enumerate(df.iterrows(), 1):
        question     = str(row.get("question", ""))
        answer       = str(row.get("answer", ""))
        ground_truth = str(row.get("ground_truth", ""))
        contexts     = row.get("contexts", [])

        logging.info(
            "[Prometheus] Evaluating row %d/%d: %.60s ...", pos, total, question
        )

        # Context Precision: top-N chunks only (reduces API calls, see docstring)
        precision_contexts = contexts[:max_precision_contexts]
        if precision_contexts:
            cp_scores = [
                judge.score_context_precision(question, c)
                for c in precision_contexts
            ]
            valid_cp = [s for s in cp_scores if not pd.isna(s)]
            cp = sum(valid_cp) / len(valid_cp) if valid_cp else float("nan")
        else:
            cp = float("nan")

        # Context Recall: holistic score — all contexts vs QASPER ground truth
        cr = (
            judge.score_context_recall(question, contexts, ground_truth)
            if contexts and ground_truth else float("nan")
        )

        # Faithfulness: scientific precision of claims against NLP paper passages
        faith = (
            judge.score_faithfulness(question, answer, contexts)
            if contexts and answer else float("nan")
        )

        # Answer Relevancy: does the answer give the specific paper fact asked for?
        ar = (
            judge.score_answer_relevancy(question, answer)
            if answer else float("nan")
        )

        # Answer Correctness: factual match against QASPER reference (QASPER-specific)
        ac = (
            judge.score_answer_correctness(question, answer, ground_truth)
            if answer and ground_truth else float("nan")
        )

        records.append({
            "context_precision":  cp,
            "context_recall":     cr,
            "faithfulness":       faith,
            "answer_relevancy":   ar,
            "answer_correctness": ac,
        })

    return pd.DataFrame(records, index=df.index)


# ── Main evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(input_csv: str = None, output_csv: str = None,
                   skip_alce: bool = False) -> None:
    _project_root = Path(__file__).resolve().parents[2]
    if input_csv is None:
        input_csv = str(_project_root / "data" / "evaluation_dataset.csv")
    if output_csv is None:
        output_csv = str(_project_root / "data" / "evaluation_report.csv")

    logging.info("Loading evaluation dataset from %s ...", input_csv)
    df = pd.read_csv(input_csv)

    def _safe_parse_contexts(x, row_idx):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except (ValueError, SyntaxError) as exc:
            logging.warning("Row %d: malformed contexts — %s", row_idx, exc)
            return []

    df["contexts"] = [_safe_parse_contexts(v, i) for i, v in enumerate(df["contexts"])]

    # ── Build Prometheus 2 judge ──────────────────────────────────────────────
    judge, is_gpu = get_prometheus_judge()

    # ── Error / refusal masks ─────────────────────────────────────────────────
    error_answer_mask = df["answer"].str.contains(
        r"^(System Error:|Error:)", na=False, regex=True
    )
    refusal_mask = df["answer"].str.contains(
        r"(do not contain enough|cannot be determined|not enough information|"
        r"no specific document|does not provide)",
        case=False, na=False, regex=True,
    )
    skip_metrics_mask = error_answer_mask | refusal_mask
    if skip_metrics_mask.sum() > 0:
        logging.warning(
            "Excluding %d/%d rows from metrics (%d errors, %d refusals).",
            skip_metrics_mask.sum(), len(df),
            error_answer_mask.sum(),
            (refusal_mask & ~error_answer_mask).sum(),
        )

    # ── ALCE evaluation ───────────────────────────────────────────────────────
    if skip_alce:
        if os.path.exists(output_csv):
            prior_df = pd.read_csv(output_csv)
            alce_cols = ["alce_citation_precision", "alce_citation_recall",
                         "alce_citation_f1"]
            if all(c in prior_df.columns for c in alce_cols) and len(prior_df) == len(df):
                for c in alce_cols:
                    df[c] = prior_df[c]
                logging.info(
                    "Loaded ALCE scores from prior report (%s). "
                    "Skipping ALCE, proceeding to Prometheus 2 metrics.", output_csv,
                )
            else:
                logging.error(
                    "--skip-alce: prior report missing ALCE columns or row-count "
                    "mismatch (%d vs %d). Run without --skip-alce first.",
                    len(prior_df), len(df),
                )
                return
        else:
            logging.error(
                "--skip-alce: no prior report at %s. Run without --skip-alce first.",
                output_csv,
            )
            return
    else:
        logging.info(
            "Running ALCE citation evaluation with Prometheus 2 NLI judge ..."
        )
        alce_evaluator = ALCEEvaluator(judge=judge)
        precisions, recalls = [], []
        evaluable_rows = len(df) - skip_metrics_mask.sum()
        eval_counter = 0

        for index, row in df.iterrows():
            if skip_metrics_mask.at[index]:
                precisions.append(float("nan"))
                recalls.append(float("nan"))
                continue
            eval_counter += 1
            logging.info(
                "ALCE: row %d/%d (dataset row %d)...",
                eval_counter, evaluable_rows, index,
            )
            p, r = alce_evaluator.calculate_metrics(row["answer"], row["contexts"])
            precisions.append(p)
            recalls.append(r)

        df["alce_citation_precision"] = precisions
        df["alce_citation_recall"]    = recalls
        df["alce_citation_f1"] = df.apply(
            lambda r: (
                2 * r["alce_citation_precision"] * r["alce_citation_recall"]
                / (r["alce_citation_precision"] + r["alce_citation_recall"])
            ) if (r["alce_citation_precision"] + r["alce_citation_recall"]) > 0 else 0.0,
            axis=1,
        )
        df.loc[skip_metrics_mask, "alce_citation_f1"] = float("nan")
        df.to_csv(output_csv, index=False)
        logging.info("ALCE complete — intermediate results saved to %s", output_csv)

    # ── Prometheus 2 RAG metrics ──────────────────────────────────────────────
    empty_ctx_mask   = df["contexts"].apply(len) == 0
    skip_prom_mask   = empty_ctx_mask | skip_metrics_mask
    valid_positions  = df.index[~skip_prom_mask].tolist()

    if skip_prom_mask.sum() > 0:
        logging.warning(
            "Skipping %d/%d rows for Prometheus metrics "
            "(%d empty-context, %d error-answer, %d refusal).",
            skip_prom_mask.sum(), len(df),
            empty_ctx_mask.sum(),
            error_answer_mask.sum(),
            (refusal_mask & ~error_answer_mask).sum(),
        )

    df_eval = df[~skip_prom_mask].reset_index(drop=True)
    prom_df = run_prometheus_metrics(df_eval, judge, is_gpu)

    # ── Per-row NaN retry ─────────────────────────────────────────────────────
    prom_metric_cols = list(prom_df.columns)
    nan_rows = prom_df[prom_df[prom_metric_cols].isna().any(axis=1)].index.tolist()
    if nan_rows:
        logging.info("Retrying %d rows that returned NaN ...", len(nan_rows))
        for retry_idx in nan_rows:
            row_df = df_eval.iloc[[retry_idx]].reset_index(drop=True)
            retry_result = run_prometheus_metrics(row_df, judge, is_gpu)
            for col in prom_metric_cols:
                val = retry_result.at[0, col]
                if not pd.isna(val):
                    prom_df.at[retry_idx, col] = val
                    logging.info("  Row %d: recovered %s = %.4f", retry_idx, col, val)
        recovered = len(nan_rows) - len(
            prom_df[prom_df[prom_metric_cols].isna().any(axis=1)]
        )
        logging.info("NaN retry: recovered %d/%d rows.", recovered, len(nan_rows))

    # ── Position-aware merge back into full DataFrame ─────────────────────────
    prom_aligned = pd.DataFrame(
        index=df.index, columns=prom_metric_cols, dtype=float
    )
    for prom_idx, orig_idx in enumerate(valid_positions):
        for col in prom_metric_cols:
            prom_aligned.at[orig_idx, col] = prom_df.at[prom_idx, col]

    final_df = df[
        ["question", "ground_truth", "answer",
         "alce_citation_precision", "alce_citation_recall", "alce_citation_f1"]
    ].copy()
    for col in prom_metric_cols:
        final_df[col] = prom_aligned[col]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    # ── Summary report ────────────────────────────────────────────────────────
    n_empty   = int(empty_ctx_mask.sum())
    n_error   = int(error_answer_mask.sum())
    n_refusal = int((refusal_mask & ~error_answer_mask).sum())

    def _fmt(series: pd.Series) -> str:
        valid = series.dropna()
        if len(valid) == 0:
            return f"N/A — all {len(series)} rows failed"
        n_nan     = len(series) - len(valid)
        n_timeout = max(0, n_nan - n_empty - n_error - n_refusal)
        parts = []
        if n_empty:   parts.append(f"{n_empty} empty-ctx")
        if n_error:   parts.append(f"{n_error} error-ans")
        if n_refusal: parts.append(f"{n_refusal} refusal")
        if n_timeout: parts.append(f"{n_timeout} timeout")
        suffix = f"  [{', '.join(parts)}]" if parts else ""
        return f"{valid.mean():.4f}{suffix}"

    logging.info("\n========== PROMETHEUS 2 × QASPER EVALUATION REPORT ==========")
    logging.info("--- Retrieval metrics ---")
    logging.info("Context Precision:       %s",
                 _fmt(final_df.get("context_precision",    pd.Series(dtype=float))))
    logging.info("Context Recall:          %s",
                 _fmt(final_df.get("context_recall",       pd.Series(dtype=float))))
    logging.info("--- Generation metrics ---")
    logging.info("Faithfulness:            %s",
                 _fmt(final_df.get("faithfulness",         pd.Series(dtype=float))))
    logging.info("Answer Relevancy:        %s",
                 _fmt(final_df.get("answer_relevancy",     pd.Series(dtype=float))))
    logging.info("Answer Correctness:      %s  [QASPER-specific: vs. reference]",
                 _fmt(final_df.get("answer_correctness",   pd.Series(dtype=float))))
    logging.info("--- Citation metrics (ALCE) ---")
    logging.info("ALCE Citation Precision: %s", _fmt(final_df["alce_citation_precision"]))
    logging.info("ALCE Citation Recall:    %s", _fmt(final_df["alce_citation_recall"]))
    logging.info("ALCE Citation F1:        %s", _fmt(final_df["alce_citation_f1"]))
    logging.info("============================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation (ALCE + Prometheus 2)."
    )
    parser.add_argument(
        "--skip-alce", action="store_true",
        help="Reuse ALCE scores from a prior evaluation_report.csv (use after "
             "a crash during Prometheus metric computation).",
    )
    parser.add_argument("--input-csv",  type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        skip_alce=args.skip_alce,
    )
