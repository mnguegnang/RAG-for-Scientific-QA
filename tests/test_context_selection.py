from src.retrieval.context_selection import (
    apply_token_budget, chunk_sort_key, ranks_from_scores, rrf_fuse,
)


def test_rrf_fuse_rewards_agreement_and_ignores_missing():
    fused = rrf_fuse({"a": {"x": 1, "y": 2}, "b": {"y": 1}}, k=60)
    assert fused["y"] > fused["x"]            # ranked by both
    assert fused["x"] == 1 / 61               # only ranker a contributes


def test_ranks_from_scores_is_one_based_descending():
    assert ranks_from_scores({"a": 0.1, "b": 0.9, "c": 0.5}) == {"b": 1, "c": 2, "a": 3}


def test_chunk_sort_key_uses_position():
    docs = [{"position": 7}, {"position": 0}, {"text": "no metadata"}, {"position": 3}]
    assert [d.get("position") for d in sorted(docs, key=chunk_sort_key)] == [0, 3, 7, None]


def test_chunk_sort_key_parses_legacy_chunk_ids():
    # Indices built before `position` existed: <paper>_<section>_<para>_<sub>
    docs = [{"chunk_id": "1909.00694_2_0_0"}, {"chunk_id": "1909.00694_0_3_1"},
            {"chunk_id": "1909.00694_0_3_0"}, {"chunk_id": "1909.00694_0_10_0"}]
    assert [d["chunk_id"] for d in sorted(docs, key=chunk_sort_key)] == [
        "1909.00694_0_3_0", "1909.00694_0_3_1", "1909.00694_0_10_0", "1909.00694_2_0_0"]


def test_token_budget_keeps_best_first_and_skips_oversized():
    docs = [{"text": "a" * 400}, {"text": "b" * 4000}, {"text": "c" * 400}]
    kept = apply_token_budget(docs, max_tokens=250)      # 100 + 1000 + 100 tokens
    assert [d["text"][0] for d in kept] == ["a", "c"]


def test_token_budget_always_keeps_one_and_zero_disables():
    docs = [{"text": "x" * 10_000}, {"text": "y"}]
    assert apply_token_budget(docs, max_tokens=10)[0]["text"][0] == "x"
    assert apply_token_budget(docs, max_tokens=0) == docs
