from __future__ import annotations

from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.reranker import Reranker
from paper_explorer.search.results import SearchResult


def run_search(
    hybrid_searcher: HybridSearcher,
    query: str,
    mode: str = "semantic",
    top_k: int = 10,
    alpha: float = 0.5,
    candidate_k: int = 50,
    rerank: bool = False,
    reranker: Reranker | None = None,
) -> list[SearchResult]:

    if not rerank:
        return hybrid_searcher.search(
            query, mode=mode, top_k=top_k, alpha=alpha, candidate_k=candidate_k
        )

    if reranker is None:
        raise ValueError("reranker must be provided when rerank=True")

    effective_candidate_k = max(candidate_k, top_k)
    candidates = hybrid_searcher.search(
        query, mode=mode, top_k=effective_candidate_k, alpha=alpha, candidate_k=candidate_k
    )
    if not candidates:
        return []

    reranked = reranker.rerank(query, candidates)
    return reranked[:top_k]
