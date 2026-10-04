"""retrieve_context() with stub retriever/reranker — no models loaded."""
from src.generation.llm_generator import LocalLLMGenerator
from src.retrieval.crag_evaluator import CRAGEvaluator
from src.run_rag import ScientificRAGPipeline


class StubRetriever:
    def __init__(self, n=30, chunks_per_paper=40):
        self.n, self.chunks_per_paper, self.calls = n, chunks_per_paper, []

    def search(self, query, k=10, rrf_k=60, dense_query=None, filter_paper_id=None):
        self.calls.append(dense_query)
        # position i; both stage-1 legs prefer late chunks, ColBERT early ones
        return [{"text": f"chunk {i} " + "w" * 40, "doc_id": "p", "position": i,
                 "dense_rank": self.n - i, "sparse_rank": self.n - i} for i in range(self.n)]

    def paper_chunk_count(self, paper_id):
        return self.chunks_per_paper


class StubReranker:
    """ColBERT prefers chunks whose position is a multiple of 3."""

    def rerank(self, query, documents, top_k=10):
        for d in documents:
            d["rerank_score"] = 20.0 + (5 if d["position"] % 3 == 0 else 0) - d["position"] * 0.1
        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return ranked if top_k is None else ranked[:top_k]

    def score(self, query, texts):
        return [10.0] * len(texts)


class StubGenerator:
    def __init__(self):
        self.hyde_calls = 0

    def generate_hypothetical_answer(self, query):
        self.hyde_calls += 1
        return "hypothetical passage"


def make_pipeline(**overrides):
    p = object.__new__(ScientificRAGPipeline)
    p.retriever, p.reranker, p.generator = StubRetriever(), StubReranker(), StubGenerator()
    p.crag_evaluator = CRAGEvaluator(correct_threshold=14, ambiguous_threshold=8,
                                     mode=overrides.pop("crag_mode", "signal"),
                                     consistency_ratio=overrides.pop("consistency", 0.0))
    p.context_k, p.context_order, p.final_ranking, p.max_context_tokens = 20, "paper", "colbert", 6000
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def test_paper_order_with_ranks_attached():
    ctx = make_pipeline().retrieve_context("short q", filter_paper_id="p")
    docs = ctx["docs"]
    assert len(docs) == 20
    positions = [d["position"] for d in docs]
    assert positions == sorted(positions)                       # OP-RAG order
    assert sorted(d["rerank_rank"] for d in docs) == list(range(1, 21))
    assert docs[0]["position"] == 0 and docs[0]["rerank_rank"] == 1


def test_rank_order_and_budget_drop_lowest_ranked():
    p = make_pipeline(context_order="rank", max_context_tokens=5 * 12)   # ~12 tokens per chunk
    docs = p.retrieve_context("q", filter_paper_id="p")["docs"]
    assert [d["rerank_rank"] for d in docs] == [1, 2, 3, 4, 5]


def test_rrf_changes_ranking_with_stage1_ranks():
    colbert = make_pipeline(context_k=5).retrieve_context("q", filter_paper_id="p")["docs"]
    rrf = make_pipeline(context_k=5, final_ranking="rrf").retrieve_context("q", filter_paper_id="p")["docs"]
    assert {d["position"] for d in colbert} != {d["position"] for d in rrf}
    assert all("fused_score" in d for d in rrf)


def test_hyde_skipped_when_it_cannot_change_candidates():
    p = make_pipeline()
    p.retrieve_context("short q", filter_paper_id="p")           # 40 chunks <= 100
    assert p.generator.hyde_calls == 0
    p.retriever.chunks_per_paper = 500
    p.retrieve_context("short q", filter_paper_id="p")
    assert p.generator.hyde_calls == 1
    rrf = make_pipeline(final_ranking="rrf")
    rrf.retrieve_context("short q", filter_paper_id="p")          # dense rank is fused
    assert rrf.generator.hyde_calls == 1


def test_legacy_incorrect_falls_back_to_top5():
    p = make_pipeline(crag_mode="legacy", consistency=0.3, context_k=10)
    p.crag_evaluator.correct_threshold = p.crag_evaluator.ambiguous_threshold = 1e9
    ctx = p.retrieve_context("q", filter_paper_id="p")
    assert ctx["crag_action"] == "Incorrect" and len(ctx["docs"]) == 5


def test_prompt_has_no_ids_scores_or_single_doc_anchor():
    docs = [{"text": "Title: T. Section: A.\nalpha", "doc_id": "1909.00694", "rerank_score": 22.5},
            {"text": "Title: T. Section: B.\nbeta", "doc_id": "1909.00694", "rerank_score": 9.1}]
    prompt = LocalLLMGenerator._build_prompt(object(), "What is alpha?", docs)
    assert "[Doc 1] Title: T. Section: A.\nalpha" in prompt
    assert "[Doc 2] Title: T. Section: B.\nbeta" in prompt
    for banned in ("Source:", "relevance", "1909.00694", "Paper Focus", "single document"):
        assert banned not in prompt
    assert "What is alpha?" in prompt


def test_vllm_retries_once_with_frequency_penalty_after_length_cutoff(monkeypatch):
    import types
    import openai

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            looped = "frequency_penalty" not in kwargs
            message = types.SimpleNamespace(content="loop " * 50 if looped else
                                            "<Final Answer>\nfine [Doc 1]\n</Final Answer>")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=message, finish_reason="length" if looped else "stop")])

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    gen = object.__new__(LocalLLMGenerator)
    gen.vllm_url, gen.model_name = "http://x/v1", "m"
    answer = LocalLLMGenerator._generate_vllm(gen, "prompt")
    assert "fine [Doc 1]" in answer
    assert len(calls) == 2 and calls[1]["frequency_penalty"] == LocalLLMGenerator.LOOP_RETRY_FREQUENCY_PENALTY
