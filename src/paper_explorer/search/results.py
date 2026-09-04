"""
Structured representation of a single semantic search result.
"""
from __future__ import annotations
from dataclasses import dataclass
from paper_explorer.data.models import Paper


@dataclass
class SearchResult:
    rank: int
    paper: Paper
    score: float

    def short_abstract(self, max_chars: int = 220) -> str:
        text = self.paper.abstract.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."

    def to_display_dict(self) -> dict:
        return {
            "rank": self.rank,
            "title": self.paper.title,
            "authors": ", ".join(self.paper.authors),
            "score": f"{self.score:.3f}",
            "published": self.paper.published or "n/a",
            "arxiv_id": self.paper.paper_id,
            "url": self.paper.abstract_url,
            "abstract": self.short_abstract(),
        }