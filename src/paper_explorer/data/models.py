""" Data models for representing research papers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory = list)
    categories: list[str] = field(default_factory = list)
    published: str | None = None
    url: str | None = None
    source: str = "arxiv"
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        known = {k:v for k,v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def text_for_embedding(self) -> str:
        return f"{self.title}\n\n{self.abstract}"

    def __repr__(self) -> str:
        return f"Paper(paper_id = {self.paper_id!r}, title = {self.title[:50]!r}...)"
