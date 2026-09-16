from __future__ import annotations

from paper_explorer.data.storage import PaperStore
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.results import SearchResult
from paper_explorer.search.searcher import PaperSearcher

VALID_MODES = ("semantic", "keyword", "hybrid")


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        fill = 1.0 if hi > 0 else 0.0
        return {k: fill for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridSearcher:
    def __init__(
        self,
        store: PaperStore,
        bm25_index: BM25Index,
        searcher: PaperSearcher | None = None,
    ):
        self.store = store
        self.bm25_index = bm25_index
        self.searcher = searcher

    def search(
        self,
        query: str,
        mode: str = "semantic",
        top_k: int = 10,
        alpha: float = 0.5,
        candidate_k: int = 50,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("Search query must not be empty")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        if len(self.store) == 0:
            return []

        if mode == "semantic":
            return self._search_semantic(query, top_k)
        if mode == "keyword":
            return self._search_keyword(query, top_k)
        return self._search_hybrid(query, top_k, alpha, candidate_k)

    def _search_semantic(self, query: str, top_k: int) -> list[SearchResult]:
        if self.searcher is None:
            raise RuntimeError("Semantic search requires a PaperSearcher (FAISS index not loaded)")
        results = self.searcher.search(query, top_k=top_k)
        for result in results:
            result.semantic_score = result.score
        return results

    def _search_keyword(self, query: str, top_k: int) -> list[SearchResult]:
        hits = self.bm25_index.search(query, top_k=top_k)
        results = []
        for rank, (paper_id, score) in enumerate(hits, start=1):
            paper = self.store.get(paper_id)
            if paper is None:
                continue
            results.append(SearchResult(rank=rank, paper=paper, score=score, keyword_score=score))
        return results

    def _search_hybrid(
        self, query: str, top_k: int, alpha: float, candidate_k: int
    ) -> list[SearchResult]:
        if self.searcher is None:
            raise RuntimeError("Hybrid search requires a PaperSearcher (FAISS index not loaded)")

        semantic_hits = self.searcher.search(query, top_k=candidate_k)
        keyword_hits = self.bm25_index.search(query, top_k=candidate_k)

        semantic_scores = {r.paper.paper_id: r.score for r in semantic_hits}
        keyword_scores = {paper_id: score for paper_id, score in keyword_hits}

        semantic_norm = min_max_normalize(semantic_scores)
        keyword_norm = min_max_normalize(keyword_scores)

        candidate_ids = sorted(set(semantic_scores) | set(keyword_scores))

        scored = []
        for paper_id in candidate_ids:
            paper = self.store.get(paper_id)
            if paper is None:
                continue
            sem = semantic_norm.get(paper_id, 0.0)
            key = keyword_norm.get(paper_id, 0.0)
            hybrid_score = alpha * sem + (1 - alpha) * key
            scored.append((paper, sem, key, hybrid_score))

        scored.sort(key=lambda item: (-item[3], item[0].paper_id))
        scored = scored[:top_k]

        return [
            SearchResult(
                rank=rank,
                paper=paper,
                score=hybrid_score,
                semantic_score=sem,
                keyword_score=key,
                hybrid_score=hybrid_score,
            )
            for rank, (paper, sem, key, hybrid_score) in enumerate(scored, start=1)
        ]
