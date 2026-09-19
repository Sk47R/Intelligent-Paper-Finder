from __future__ import annotations

from dataclasses import dataclass

from paper_explorer.data.models import Paper


@dataclass
class SearchResult:
    rank: int
    paper: Paper
    score: float
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None

    def short_abstract(self, max_chars: int = 220) -> str:
        text = self.paper.abstract.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."

    def to_display_dict(self) -> dict:
        d = {
            "rank": self.rank,
            "title": self.paper.title,
            "authors": ", ".join(self.paper.authors),
            "score": f"{self.score:.3f}",
            "published": self.paper.published or "n/a",
            "arxiv_id": self.paper.paper_id,
            "url": self.paper.abstract_url,
            "abstract": self.short_abstract(),
        }
        if self.semantic_score is not None:
            d["semantic_score"] = f"{self.semantic_score:.3f}"
        if self.keyword_score is not None:
            d["keyword_score"] = f"{self.keyword_score:.3f}"
        if self.hybrid_score is not None:
            d["hybrid_score"] = f"{self.hybrid_score:.3f}"
        if self.rerank_score is not None:
            d["rerank_score"] = f"{self.rerank_score:.3f}"
        return d
