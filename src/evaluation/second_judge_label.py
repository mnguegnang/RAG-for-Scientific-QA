"""
Second-judge auto-labeling — fills `human_label` in data/judge_validation_blind.csv
using Claude as an independent, stronger judge (key_metrics_improvements.md,
Finding #4: "have a human — or a second, independent, stronger judge such as
Claude or GPT-4 at temperature 0 — label context_recall/faithfulness/
ALCE-entailment for each").

Design notes:
    - One stateless API call per atomic unit, thinking disabled (Sonnet 5/
      Opus 5/Haiku 4.5 no longer accept `temperature`/`top_p`/`top_k` at
      all — 400 invalid_request_error — so determinism for this one-word
      True/False task comes from skipping the reasoning trace, not from a
      sampling parameter). No project context is given beyond what
      validate_judge.py already put in the blind file (unit_text +
      context_shown) — matching how Prometheus
      itself is called, and keeping this judge blind to Prometheus's
      decision (which lives only in judge_validation_key.csv, never read
      here) to avoid anchoring bias.
    - Each metric's question is phrased identically to the corresponding
      Prometheus prompt in evaluate_rag.py (_score_context_recall_items,
      _score_faithfulness_items, check_nli_entailment) so the two judges are
      answering the same substantive question — only the model differs.
    - Writes back into data/judge_validation_blind.csv in place by default
      (the file compute_agreement.py already expects), or to --output if
      given.

Setup:
    echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
    uv pip install --python .venv/bin/python anthropic

Usage:
    python -m src.evaluation.second_judge_label
    python -m src.evaluation.second_judge_label --model claude-haiku-4-5
    python -m src.evaluation.second_judge_label --blind-csv data/judge_validation_blind.csv --output data/judge_validation_blind.csv
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-sonnet-5"
MAX_RETRIES = 3

_PROMPTS = {
    "context_recall": (
        "You are a scientific RAG evaluator.\n"
        "Retrieved passages:\n{context_shown}\n\n"
        "Ground-truth sentence: {unit_text}\n\n"
        "Does at least one retrieved passage directly support this "
        "ground-truth sentence? "
        "Answer with exactly one word: True or False."
    ),
    "faithfulness": (
        "Retrieved passages:\n{context_shown}\n\n"
        "Claim from generated answer: {unit_text}\n\n"
        "Is this claim directly supported by at least one of the "
        "retrieved passages? "
        "Answer with exactly one word: True or False."
    ),
    "alce_entailment": (
        "You are an NLI judge for scientific text. "
        "Determine whether the document passage below supports the claim. "
        "Paraphrasing and implicit support both count; exact wording is not "
        "required. For numerical claims, percentage and decimal formats are "
        "equivalent ('84.3%' and '0.843' express the same value).\n\n"
        "Document passage:\n{context_shown}\n\n"
        "Claim: {unit_text}\n\n"
        "Does the document passage support the claim? "
        "Answer with exactly one word: True or False."
    ),
}


def _parse_true_false(text: str) -> Optional[bool]:
    """Same parsing convention as evaluate_rag.py's atomic helpers."""
    lowered = text.strip().lower()
    if "true" in lowered:
        return True
    if "false" in lowered:
        return False
    if "yes" in lowered:
        return True
    if "no" in lowered:
        return False
    return None


def label_row(client, model: str, metric: str, unit_text: str, context_shown: str) -> Optional[bool]:
    template = _PROMPTS.get(metric)
    if template is None:
        logging.warning("Unknown metric %r — leaving unlabeled.", metric)
        return None

    prompt = template.format(context_shown=str(context_shown)[:6000], unit_text=str(unit_text)[:1000])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=16,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            result = _parse_true_false(text)
            if result is None:
                logging.warning("Could not parse True/False from: %.80s", text)
            return result
        except Exception as exc:
            wait = 2 ** attempt
            logging.warning(
                "API call failed (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    logging.error("Giving up on this unit after %d retries.", MAX_RETRIES)
    return None


def run_labeling(
    blind_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Put it in a .env file at the project "
            "root (ANTHROPIC_API_KEY=sk-ant-...) or export it in your shell."
        )

    import anthropic
    client = anthropic.Anthropic()

    blind_csv = blind_csv or str(_PROJECT_ROOT / "data" / "judge_validation_blind.csv")
    output_csv = output_csv or blind_csv

    logging.info("Loading %s ...", blind_csv)
    df = pd.read_csv(blind_csv)
    df["human_label"] = df["human_label"].astype(object)

    existing = df["human_label"].astype(str).str.strip().replace("nan", "")
    todo_mask = existing.eq("")
    n_todo = int(todo_mask.sum())
    logging.info("%d/%d units already labeled — labeling the remaining %d.",
                 len(df) - n_todo, len(df), n_todo)

    for pos, idx in enumerate(df.index[todo_mask], start=1):
        row = df.loc[idx]
        result = label_row(client, model, row["metric"], row["unit_text"], row["context_shown"])
        df.at[idx, "human_label"] = "" if result is None else str(result)
        if pos % 10 == 0 or pos == n_todo:
            logging.info("[%d/%d] labeled (unit_id=%s -> %s)", pos, n_todo, row["unit_id"], result)

    df.to_csv(output_csv, index=False)
    logging.info("Wrote labels to %s", output_csv)
    logging.info("Next: python -m src.evaluation.compute_agreement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label data/judge_validation_blind.csv using Claude as an "
                     "independent second judge (Finding #4)."
    )
    parser.add_argument("--blind-csv", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_labeling(blind_csv=args.blind_csv, output_csv=args.output, model=args.model)
