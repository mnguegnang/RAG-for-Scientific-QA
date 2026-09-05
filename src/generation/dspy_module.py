"""
DSPy-compiled generation module (key_metrics_improvements.md, Finding #5).

Ports the hand-written prompt in configs/prompts.yaml (`rag_prompt_template`)
into a dspy.Signature + dspy.ChainOfThought module, so a compiler
(src/evaluation/compile_dspy_prompt.py) can optimize the few-shot
demonstrations against a citation-format/anti-stuffing metric instead of
hand-tuning the prompt text.

This is a *direct port*, not a redesign: the citation-tag rules and
paper-focus instruction are the same ones already in configs/prompts.yaml,
carried here as the Signature's field descriptions/docstring so the compiled
program produces output in the same shape. `render_tagged_output` re-wraps
the module's typed (reasoning, cited_answer) output into the exact
`<Reasoning>...</Reasoning><Final Answer>...</Final Answer>` string shape
`generate_predictions.py::extract_final_answer` already parses, so nothing
downstream (ALCE, Prometheus scoring, extract_final_answer) needs to change
to consume DSPy-generated answers.

Note: dspy.ChainOfThought automatically prepends a `reasoning` output field
to whatever signature it wraps (see dspy.ChainOfThought source) — the
Signature below deliberately does NOT declare `reasoning` itself, only
`cited_answer`, to avoid a duplicate-field conflict.
"""
import re
import sys
from typing import Any, Dict, List

# litellm 1.99.0 (pulled in by dspy) declares Requires-Python >=3.10 but
# src/litellm/llms/anthropic/experimental_pass_through/context_management/
# editors/compact.py unconditionally does `from typing import ... NotRequired`
# — NotRequired is Python 3.11+ stdlib only (on 3.10 it lives in
# typing_extensions). This breaks bare `import litellm` on this project's
# pinned Python 3.10 (CLAUDE.md), which surfaces here as a confusing
# ImportError deep inside dspy.LM's lazy litellm import, on the *first* real
# (non-cached) generation call — not at import time. No litellm version fixes
# this (1.99.0 is latest; upgrading its transitive deps risks the
# huggingface_hub>=1.0/adapters conflict CLAUDE.md also warns about), so patch
# the stdlib module before dspy/litellm ever gets a chance to import it. Both
# DSPy entry points (this module and llm_generator.py's DSPY_COMPILED_PROMPT_PATH
# path) import this file before making any real dspy.LM call.
if sys.version_info < (3, 11):
    import typing as _typing
    if not hasattr(_typing, "NotRequired"):
        from typing_extensions import NotRequired as _NotRequired
        _typing.NotRequired = _NotRequired

import dspy

# Second, independent litellm 1.99.0 bug hit right after the one above: its
# litellm.types.utils.Message (built with a forward-referenced
# `ChatCompletionReasoningSummaryTextBlock` type) is never rebuilt with that
# name in scope, so the *first* real ModelResponse()/Choices()/Message()
# construction inside litellm.completion() raises
# `PydanticUserError: Message is not fully defined`. Rebuilding once here
# (idempotent) resolves the whole nested chain — reproduced directly via
# `litellm.types.utils.ModelResponse()` with no vLLM/GPU call needed.
from litellm.types.llms.openai import ChatCompletionReasoningSummaryTextBlock as _CRSTB
import litellm.types.utils as _litellm_types_utils
_litellm_types_utils.Message.model_rebuild(
    _types_namespace={"ChatCompletionReasoningSummaryTextBlock": _CRSTB}
)

# Same citation-format rules as configs/prompts.yaml's <Instructions> block —
# kept in sync manually; if the YAML template changes, update this docstring.
_CITATION_RULES = (
    "You are a precise scientific AI research assistant. Answer the "
    "question based ONLY on the provided context.\n"
    "1. Comprehension: read the context carefully. If it does not contain "
    "the answer, reply exactly with: \"The retrieved documents do not "
    "contain enough information to answer this.\" Do not guess.\n"
    "2. Citations: every factual claim MUST end with EXACTLY the tag "
    "[Doc N] where N is the document number (e.g. [Doc 1], [Doc 2]). Do "
    "NOT use any other format such as (Doc 1), [Document 1], or [1]."
)

DEFAULT_PAPER_FOCUS_HINT = (
    "First, identify which single document is most directly relevant to "
    "the question. Anchor your answer primarily to that document. Mention "
    "other documents only if they add genuinely complementary information."
)


class ScientificRAGAnswer(dspy.Signature):
    __doc__ = _CITATION_RULES

    context: List[str] = dspy.InputField(
        desc="Numbered retrieved passages, one per list item, each already "
             "labelled '[Doc N] Source: ... Content: ...'."
    )
    paper_focus_hint: str = dspy.InputField(
        desc="Instruction for which single document to anchor the answer to."
    )
    question: str = dspy.InputField(desc="The user's scientific question.")
    cited_answer: str = dspy.OutputField(
        desc="The final, cited answer. Every factual claim ends with "
             "exactly one or more [Doc N] tags."
    )


class ScientificRAGModule(dspy.Module):
    """dspy.ChainOfThought over ScientificRAGAnswer — adds a `reasoning`
    output field automatically (see module docstring)."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ScientificRAGAnswer)

    def forward(self, context: List[str], paper_focus_hint: str, question: str):
        return self.generate(
            context=context, paper_focus_hint=paper_focus_hint, question=question
        )


def format_context_blocks(retrieved_docs: List[Dict[str, Any]]) -> List[str]:
    """
    One string per retrieved doc, matching LocalLLMGenerator._build_prompt's
    per-doc block format exactly (src/generation/llm_generator.py:195-201) —
    same [Doc N]/Source/Content shape whether generation goes through the
    static template or this DSPy module, so [Doc N] citation indices map to
    `contexts[N-1]` identically either way.
    """
    blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        doc_id = doc.get("doc_id", doc.get("id", f"doc_{i}"))
        score = doc.get("rerank_score", doc.get("score", None))
        score_str = f"  [relevance: {score:.4f}]" if score is not None else ""
        blocks.append(
            f"[Doc {i}]\nSource: {doc_id}{score_str}\nContent: {doc.get('text', '')}"
        )
    return blocks


def render_tagged_output(reasoning: str, cited_answer: str) -> str:
    """
    Re-wrap DSPy's typed output into the exact tag shape
    extract_final_answer (generate_predictions.py) already parses:
    <Reasoning>...</Reasoning><Final Answer>...</Final Answer>.
    """
    return (
        f"<Reasoning>\n{reasoning.strip()}\n</Reasoning>\n"
        f"<Final Answer>\n{cited_answer.strip()}\n</Final Answer>"
    )


if __name__ == "__main__":
    # ── Smoke test: verifies Signature/module wiring and the render/parse
    # round-trip WITHOUT any live LM call (dspy.utils.dummies.DummyLM stubs
    # the completion). No vLLM server, no GPU, no network needed.
    from dspy.utils.dummies import DummyLM

    stub_lm = DummyLM([
        {"reasoning": "Doc 1 states the fact directly.", "cited_answer": "The answer is X [Doc 1]."},
    ])
    dspy.settings.configure(lm=stub_lm)

    module = ScientificRAGModule()
    retrieved_docs = [
        {"doc_id": "1909.00694", "rerank_score": 22.5, "text": "The seed lexicon consists of positive and negative predicates."},
        {"doc_id": "1909.00695", "rerank_score": 10.1, "text": "Unrelated passage about a different topic."},
    ]
    pred = module(
        context=format_context_blocks(retrieved_docs),
        paper_focus_hint=DEFAULT_PAPER_FOCUS_HINT,
        question="What is the seed lexicon?",
    )
    assert hasattr(pred, "reasoning"), "ChainOfThought did not produce a `reasoning` field"
    assert hasattr(pred, "cited_answer"), "Signature did not produce a `cited_answer` field"
    print(f"reasoning={pred.reasoning!r}")
    print(f"cited_answer={pred.cited_answer!r}")

    rendered = render_tagged_output(pred.reasoning, pred.cited_answer)
    print("\n--- rendered ---")
    print(rendered)

    # Round-trip through the exact regex generate_predictions.py uses.
    m = re.search(
        r'<Final\s+Answer>\s*(.*?)\s*(?:</Final\s+Answer>|(?=<Final\s+Answer>)|$)',
        rendered, re.DOTALL | re.IGNORECASE,
    )
    assert m and m.group(1) == pred.cited_answer.strip(), (
        "render_tagged_output output did not round-trip through "
        "extract_final_answer's regex"
    )
    print("\nSMOKE TEST PASSED")
