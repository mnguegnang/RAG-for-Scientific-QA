from src.evaluation.evaluate_rag import PrometheusJudge, precision_contexts
from src.evaluation.evaluate_retrieval import evidence_found, score_question
from src.evaluation.grounding import GroundingJudge, chunk_document


class KeywordJudge(GroundingJudge):
    """Supported iff the claim's last word appears in some passage."""

    def __init__(self):
        self.calls = []

    def _supported(self, claim, passages, kind):
        self.calls.append((claim, len(passages), kind))
        word = claim.rstrip(".").split()[-1]
        return any(word in p for p in passages), None


def test_faithfulness_sees_every_context():
    contexts = [f"passage {i}" for i in range(10)] + ["the answer is quokka"]
    judge = KeywordJudge()
    score = judge.score_faithfulness("q", "The animal is a quokka [Doc 11].", contexts)
    assert score == 1.0                       # supported only by the 11th context
    assert len(judge.calls) == 1 and judge.calls[0][1:] == (11, "faithfulness")
    assert "[Doc" not in judge.calls[0][0]


def test_nli_strips_citation_tags_and_short_circuits():
    judge = KeywordJudge()
    assert judge.check_nli_entailment("[Doc 2]", "text") == (False, None)
    assert judge.check_nli_entailment("x is quokka [Doc 1][Doc 2]", "") == (False, None)
    assert judge.check_nli_entailment("x is quokka [Doc 1][Doc 2]", "a quokka")[0] is True


def test_precision_contexts_uses_ranks_not_position():
    contexts = ["intro", "method", "results", "table"]
    assert precision_contexts(contexts, [4, 2, 1, 3], 3) == ["results", "method", "table"]
    assert precision_contexts(contexts, None, 2) == ["intro", "method"]          # legacy rows
    assert precision_contexts(contexts, [1, 2], 2) == ["intro", "method"]        # misaligned


def test_prometheus_passage_batches_respect_budget_and_numbering():
    passages = ["x" * 1400] * 15
    batches = PrometheusJudge._passage_batches(passages)
    assert len(batches) > 1
    assert all(len(b) <= PrometheusJudge.PASSAGE_BUDGET_CHARS + 100 for b in batches)
    assert "[Passage 15]" in batches[-1] and "[Passage 1]" in batches[0]


def test_chunk_document_splits_on_sentences():
    doc = " ".join(f"Sentence number {i} has five words." for i in range(300))
    chunks = chunk_document(doc, chunk_words=500)
    assert len(chunks) == 4 and all(len(c.split()) <= 500 for c in chunks)
    assert chunk_document("") == [""]


def test_evidence_matching_text_split_and_float():
    para = "We train the model with a learning rate of 0.001 for ten epochs. " * 5
    contexts = [f"Title: P. Section: Setup.\n{para}",
                "Title: P. Section: Table 3.\nTable 3: Performance of various models on the ACP test set.\n| a |"]
    assert evidence_found(para, contexts)
    # a later piece of a split paragraph still counts
    assert evidence_found("Intro sentence. " + para, ["Title: P. Section: S.\n" + para])
    assert evidence_found("FLOAT SELECTED: Table 3: Performance of various models on the ACP test set.",
                          contexts)
    assert not evidence_found("FLOAT SELECTED: Table 9: Missing.", contexts)
    s = score_question([para, "FLOAT SELECTED: Table 9: Missing.", ""], contexts)
    assert s["text_recall"] == 1.0 and s["float_hit"] == 0.0 and s["all_recall"] == 0.5


def test_true_false_parser_reads_prose_verdicts():
    parse = PrometheusJudge._parse_true_false
    assert parse("True", "x") is True and parse("false.", "x") is False
    assert parse("The ground-truth sentence is directly supported by Passage 3, which", "x") is True
    assert parse("Passage 2 supports the claim.", "x") is True
    assert parse("The claim is not supported by any passage.", "x") is False
    assert parse("This passage doesn't support it", "x") is False
    assert parse("The sentence is unsupported.", "x") is False
    assert parse("The ground-truth sentence is missing, so it is impossible to determine", "x") is False


def test_compare_reports_pairs_on_common_questions():
    import pandas as pd
    from src.evaluation.compare_reports import compare
    a = pd.DataFrame({"question": ["q1", "q2", "q3"], "answer_correctness": [0.0, 0.5, None]})
    b = pd.DataFrame({"question": ["q1", "q2", "q3"], "answer_correctness": [1.0, 1.0, 1.0]})
    row = compare(a, b).iloc[0]
    assert row["A_n"] == 2 and row["B_n"] == 3 and row["paired_n"] == 2
    assert row["paired_delta"] == 0.75 and row["resolved"]


def test_minicheck_batches_respect_attention_budget():
    from src.evaluation.grounding import MiniCheckJudge
    judge = object.__new__(MiniCheckJudge)
    judge.batch_size = 16
    lengths = [2048] * 5 + [700] * 20
    batches = list(judge._batches(lengths))
    assert sorted(i for b in batches for i in b) == list(range(25))        # every input once
    for b in batches:
        longest = max(lengths[i] for i in b)
        assert len(b) <= 16 and (len(b) == 1 or len(b) * longest ** 2 <= judge.ATTENTION_BUDGET)
    assert max(len(b) for b in batches if lengths[b[0]] == 2048) == 2    # long inputs: 2 at a time
