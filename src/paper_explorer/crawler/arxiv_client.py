from __future__ import annotations
import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
import feedparser
import requests
from paper_explorer.data.models import Paper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"


class ArxivAPIError(Exception):
    """Raised when the arXiv API cannot be reached after retries."""


class ArxivClient:
    def __init__(
        self,
        page_size: int = 100,
        delay_seconds: float = 3.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
    ):
        self.page_size = page_size
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def search(
        self,
        query: str,
        max_results: int = 100,
        start: int = 0,
        category: str | None = None,
        raw_dir: str | Path | None = None,
    ) -> list[Paper]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if max_results <= 0:
            raise ValueError("max_results must be positive")

        search_query = f"all:{query}"
        if category:
            search_query = f"({search_query}) AND cat:{category}"

        papers: list[Paper] = []
        fetched = 0
        page_num = 0
        while fetched < max_results:
            batch = min(self.page_size, max_results - fetched)
            params = {
                "search_query": search_query,
                "start": start + fetched,
                "max_results": batch,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            raw_text = self._get_with_retries(params)
            if raw_dir:
                self._save_raw(raw_dir, query, page_num, raw_text)

            feed = feedparser.parse(raw_text)
            
            if not feed.entries:
                break
            
            papers.extend(self._entry_to_paper(entry) for entry in feed.entries)
            fetched += len(feed.entries)
            page_num += 1
            
            if len(feed.entries) < batch:
                break
            
            time.sleep(self.delay_seconds)

        logger.info("Fetched %d papers for query=%r category=%r", len(papers), query, category)
        return papers

    def _get_with_retries(self, params: dict) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    ARXIV_API_URL, params = params, timeout = self.timeout_seconds
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "arXiv API request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
        raise ArxivAPIError(
            f"arXiv API request failed after {self.max_retries} attempts"
        ) from last_error

    @staticmethod
    def _save_raw(raw_dir: str | Path, query: str, page_num: int, raw_text: str) -> None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents = True, exist_ok = True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower())[:50]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = raw_dir / f"{timestamp}_{slug}_page{page_num}.xml"
        path.write_text(raw_text, encoding = "utf-8")
        logger.debug("Saved raw arXiv response to %s", path)

    @staticmethod
    def _entry_to_paper(entry) -> Paper:
        arxiv_id = entry.id.split("/abs/")[-1]
        categories = [tag["term"] for tag in entry.get("tags", [])]
        pdf_url = next(
            (
                link.get("href")
                for link in entry.get("links", [])
                if link.get("type") == "application/pdf"
            ),
            None,
        )
        return Paper(
            paper_id = arxiv_id,
            title = " ".join(entry.title.split()),
            abstract = " ".join(entry.summary.split()),
            authors = [author.name for author in entry.get("authors", [])],
            categories = categories,
            published = (entry.get("published") or "")[:10] or None,
            updated = (entry.get("updated") or "")[:10] or None,
            pdf_url = pdf_url,
            abstract_url = entry.get("link"),
            source = "arxiv",
        )