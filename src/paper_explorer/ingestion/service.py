from __future__ import annotations

import logging

import numpy as np

from paper_explorer.crawler.arxiv_client import ArxivClient
from paper_explorer.data.repository import PaperRepository
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.index import VectorIndex

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        repository: PaperRepository,
        client: ArxivClient | None = None,
        embedding_model: EmbeddingModel | None = None,
        embed_batch_size: int = 32,
    ):
        self.repository = repository
        self.client = client or ArxivClient()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.embed_batch_size = embed_batch_size

    def ingest(
        self,
        query: str,
        max_results: int = 100,
        start: int = 0,
        category: str | None = None,
        raw_dir=None,
    ) -> dict:
        papers = self.client.search(
            query, max_results=max_results, start=start, category=category, raw_dir=raw_dir
        )

        new_count = changed_count = unchanged_count = 0
        for paper in papers:
            status = self.repository.upsert_paper(paper)
            if status == "new":
                new_count += 1
            elif status == "changed":
                changed_count += 1
            else:
                unchanged_count += 1

        embedded_count, failed_count = self._embed_pending()

        summary = {
            "fetched": len(papers),
            "new": new_count,
            "changed": changed_count,
            "unchanged": unchanged_count,
            "embedded": embedded_count,
            "embedding_failed": failed_count,
        }
        logger.info("Ingestion summary: %s", summary)
        return summary

    def _embed_pending(self) -> tuple[int, int]:
        pending = self.repository.get_papers_needing_embedding()
        if not pending:
            return 0, 0

        embedded = 0
        failed = 0
        for i in range(0, len(pending), self.embed_batch_size):
            batch = pending[i : i + self.embed_batch_size]
            texts = [p.text_for_embedding for p in batch]
            try:
                vectors = self.embedding_model.embed_texts(texts)
            except Exception:
                logger.exception("Embedding batch failed; marking %d papers as failed", len(batch))
                for paper in batch:
                    self.repository.mark_embedding_failed(paper.paper_id)
                failed += len(batch)
                continue

            for paper, vector in zip(batch, vectors):
                self.repository.set_embedding(
                    paper.paper_id, vector, self.embedding_model.model_name
                )
                embedded += 1
        return embedded, failed

    def rebuild_index(self) -> VectorIndex:
        pairs = self.repository.get_all_embedded_vectors()
        index = VectorIndex()
        if pairs:
            paper_ids = [pid for pid, _ in pairs]
            matrix = np.array([vec for _, vec in pairs], dtype=np.float32)
            index.build(paper_ids, matrix)
        return index
