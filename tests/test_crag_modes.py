import pytest

from src.retrieval.crag_evaluator import CRAGEvaluator


def _docs():
    return [
        {"text": "Title: P. Section: Results.\nWe report 0.843 accuracy. The baseline gets 0.5. "
                 "Training took two days. The model is a BiGRU.", "rerank_score": 10.0},
        {"text": "Title: P. Section: Method.\nWe use a seed lexicon.", "rerank_score": 20.0},
        {"text": "Title: P. Section: Misc.\nUnrelated text here about other things.",
         "rerank_score": 2.0},
    ]


def test_signal_mode_never_drops_or_rewrites():
    crag = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8, mode="signal")
    docs = _docs()
    action, refined, details = crag.evaluate_and_refine("what accuracy?", docs)
    assert action == "Correct"                  # one doc >= upper threshold (CRAG §4.3)
    assert [d["text"] for d in refined] == [d["text"] for d in _docs()]
    assert details["mode"] == "signal"


def test_default_action_rule_needs_only_one_correct_doc():
    crag = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8)
    docs = [{"rerank_score": 15.0}] + [{"rerank_score": 1.0}] * 9
    assert crag.evaluate_and_refine("q", docs)[0] == "Correct"
    legacy = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8,
                           consistency_ratio=0.3, mode="legacy")
    assert legacy.evaluate_and_refine("q", [dict(d) for d in docs])[0] == "Ambiguous"


def test_legacy_mode_drops_incorrect_docs():
    crag = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8,
                         consistency_ratio=0.3, mode="legacy")
    _, refined, _ = crag.evaluate_and_refine("accuracy baseline model", _docs())
    assert all(d["rerank_score"] >= 8 for d in refined)


def test_refine_mode_scores_strips_and_keeps_order_and_header():
    def scorer(query, strips):
        return [20.0 if "0.843" in s or "BiGRU" in s else 1.0 for s in strips]

    crag = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8,
                         mode="refine", strip_scorer=scorer)
    action, refined, _ = crag.evaluate_and_refine("what accuracy?", _docs())
    assert len(refined) == 3                                 # nothing dropped
    ambiguous = refined[0]
    assert ambiguous["text"].startswith("Title: P. Section: Results.\n")
    assert ambiguous["text"].endswith("We report 0.843 accuracy. The model is a BiGRU.")
    assert ambiguous["refined"] is True
    assert refined[2]["text"] == _docs()[2]["text"]          # Incorrect kept intact


def test_refine_mode_without_scorer_keeps_documents():
    crag = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8, mode="refine")
    _, refined, _ = crag.evaluate_and_refine("q", _docs())
    assert [d["text"] for d in refined] == [d["text"] for d in _docs()]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        CRAGEvaluator(mode="filter")
