from __future__ import annotations

import time
from collections.abc import Iterator

import feedparser
import requests

from paper_explorer.data.models import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"

class ArxivCrawler:
    def __init__(self, page_size: int = 100, delays_seconds: float = 3.0):
        self.page_size = page_size
        self.delay_seconds = delays_seconds

    def search(self, query: str, max_results: int = 100) -> list[Paper]:
        return [self._entry_to_paper(entry) for entry in self._iter_entries(query, max_results)]

    def _iter_entries(self, query: str, max_results: int) -> Iterator[dict]:
        fetched = 0
        while fetched < max_results:
            batch = min(self.page_size, max_results - fetched)
            params = {
                "search_query": f"all:{query}",
                "start": fetched,
                "max_results": batch,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            response = requests.get(ARXIV_API_URL, params = params, timeout = 30)
            response.raise_for_status()
            feed = feedparser.parse(response.text)

            if not feed.entries:
                break
            yield from feed.entries
            fetched += len(feed.entries)

            if len(feed.entries) < batch:
                break
            time.sleep(self.delay_seconds)


    @staticmethod
    def _entry_to_paper(entry) -> Paper:
        arxiv_id = entry.id.split("/abs/")[-1]
        categories = [tag["term"] for tag in entry.get("tags", [])]

        return Paper(
            paper_id = arxiv_id,
            title = " ".join(entry.title.split()),
            abstract = " ".join(entry.summary.split()),
            authors = [author.name for author in entry.get("authors", [])],
            categories = categories,
            published = (entry.get("published") or "")[:10] or None,
            url = entry.get("link"),
            source = "arxiv",
        )


