from paper_explorer.crawler.arxiv_client import ArxivAPIError, ArxivClient
from paper_explorer.data.content_hash import compute_content_hash
from paper_explorer.data.models import Paper
from paper_explorer.data.repository import PaperRepository
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.ingestion.service import IngestionService
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.pipeline import run_search
from paper_explorer.search.reranker import CrossEncoderReranker, Reranker, RerankerLoadError
from paper_explorer.search.results import SearchResult
from paper_explorer.search.searcher import PaperSearcher

__all__ = [
    "Paper",
    "PaperStore",
    "PaperRepository",
    "compute_content_hash",
    "IngestionService",
    "ArxivClient",
    "ArxivAPIError",
    "EmbeddingModel",
    "VectorIndex",
    "SearchResult",
    "PaperSearcher",
    "BM25Index",
    "HybridSearcher",
    "run_search",
    "Reranker",
    "CrossEncoderReranker",
    "RerankerLoadError",
]

__version__ = "0.4.0"
