from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from paper_explorer.search.results import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerLoadError(Exception):
    """Raised when a Reranker's underlying model fails to load."""


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 16,
        model=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        if model is not None:
            self._model = model
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker %r ...", model_name)
            self._model = CrossEncoder(model_name)
        except Exception as exc:
            raise RerankerLoadError(
                f"Failed to load cross-encoder model {model_name!r}: {exc}"
            ) from exc

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, result.paper.text_for_embedding) for result in candidates]

        import torch

        with torch.inference_mode():
            scores = self._model.predict(pairs, batch_size=self.batch_size)

        for result, score in zip(candidates, scores):
            result.rerank_score = float(score)

        candidates.sort(key=lambda r: (-r.rerank_score, r.paper.paper_id))
        for rank, result in enumerate(candidates, start=1):
            result.rank = rank
        return candidates
