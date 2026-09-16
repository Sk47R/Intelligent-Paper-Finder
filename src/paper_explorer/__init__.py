from paper_explorer.crawler.arxiv_client import ArxivAPIError, ArxivClient
from paper_explorer.data.models import Paper
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.results import SearchResult
from paper_explorer.search.searcher import PaperSearcher
from paper_explorer.summarize.topic_summary import Topic, TopicSummarizer

__all__ = [
    "Paper",
    "PaperStore",
    "ArxivClient",
    "ArxivAPIError",
    "EmbeddingModel",
    "VectorIndex",
    "SearchResult",
    "PaperSearcher",
    "BM25Index",
    "HybridSearcher",
    "Topic",
    "TopicSummarizer",
]

__version__ = "0.2.0"