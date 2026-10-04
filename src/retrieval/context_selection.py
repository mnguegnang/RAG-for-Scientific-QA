"""
Pure helpers that decide *which* reranked chunks reach the generator and in
*what order*. Kept free of model code so the pipeline (run_rag.py) and the
offline retrieval harness (evaluation/evaluate_retrieval.py) share one
implementation.

  - rrf_fuse            — Reciprocal Rank Fusion over several rankers
                          (Cormack, Clarke & Büttcher, SIGIR 2009).
  - chunk_sort_key      — position of a chunk inside its paper, for
                          order-preserving RAG (Yu et al. 2024, OP-RAG,
                          arXiv:2409.01666, §3 Eq. 2).
  - apply_token_budget  — drop the lowest-ranked chunks until the context
                          fits the generator's window.
"""
from typing import Dict, Hashable, List, Optional, Tuple


def rrf_fuse(rankings: Dict[str, Dict[Hashable, int]], k: int = 60) -> Dict[Hashable, float]:
    """
    Reciprocal Rank Fusion.

    rankings: {ranker_name: {item_key: 1-based rank}}. An item missing from a
    ranker simply gets no contribution from it (the standard RRF treatment).
    Returns {item_key: fused score}; higher is better.
    """
    fused: Dict[Hashable, float] = {}
    for ranks in rankings.values():
        for key, rank in ranks.items():
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def ranks_from_scores(scores: Dict[Hashable, float]) -> Dict[Hashable, int]:
    """Turn {key: score} (higher = better) into {key: 1-based rank}."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {key: i + 1 for i, (key, _) in enumerate(ordered)}


def chunk_sort_key(doc: Dict) -> Tuple:
    """
    Sort key placing a chunk at its position in the source paper.

    Indices built by the current chunker store an integer ``position``
    (abstract first, then paragraphs in reading order, then tables/figures).
    Older indices only carry ``chunk_id = <paper>_<section>_<para>_<sub>``,
    which encodes the same order, so it is parsed as a fallback. Chunks with
    neither sort last, in their incoming order (Python's sort is stable).
    """
    position = doc.get("position")
    if position is not None:
        return (0, float(position), 0, 0)
    chunk_id = doc.get("chunk_id")
    if chunk_id:
        parts = str(chunk_id).rsplit("_", 3)
        if len(parts) == 4:
            try:
                return (0, float(parts[1]), int(parts[2]), int(parts[3]))
            except ValueError:
                pass
    return (1, 0.0, 0, 0)


def approx_tokens(text: str) -> int:
    """~4 characters per token for English scientific prose (Llama 3 BPE)."""
    return max(1, len(text) // 4)


def apply_token_budget(docs_by_rank: List[Dict], max_tokens: Optional[int]) -> List[Dict]:
    """
    Keep the highest-ranked docs whose combined size fits ``max_tokens``.

    docs_by_rank must be in rank order (best first). The best doc is always
    kept, even if it alone exceeds the budget, so the generator never gets an
    empty context because of a single long chunk.
    """
    if not max_tokens or max_tokens <= 0:
        return list(docs_by_rank)
    kept, used = [], 0
    for doc in docs_by_rank:
        cost = approx_tokens(doc.get("text", ""))
        if kept and used + cost > max_tokens:
            continue
        kept.append(doc)
        used += cost
    return kept
