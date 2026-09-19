from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from paper_explorer.config import (
    DEFAULT_ID_MAP_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_STORE_PATH,
    RAW_DIR,
)
from paper_explorer.crawler.arxiv_client import ArxivAPIError, ArxivClient
from paper_explorer.data.models import Paper
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.pipeline import run_search
from paper_explorer.search.reranker import CrossEncoderReranker, RerankerLoadError
from paper_explorer.search.results import SearchResult
from paper_explorer.search.searcher import PaperSearcher

console = Console()
logger = logging.getLogger("paper_explorer")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
        force=True,
    )


def _load_vector_index(args: argparse.Namespace) -> VectorIndex:
    try:
        return VectorIndex.load(args.index, args.id_map)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(f"[red]No FAISS index found at {args.index}. Run `ingest` first.[/red]")
        raise SystemExit(1) from exc


def cmd_ingest(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    client = ArxivClient()

    console.print(f"[bold]Querying arXiv for[/bold] {args.query!r} ...")
    try:
        papers = client.search(
            args.query,
            max_results=args.max_results,
            start=args.start,
            category=args.category,
            raw_dir=None if args.no_raw else args.raw_dir,
        )
    except ArxivAPIError as exc:
        console.print(f"[red]arXiv API error:[/red] {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        console.print(f"[red]Invalid arguments:[/red] {exc}")
        raise SystemExit(1) from exc

    if not papers:
        console.print("[yellow]No papers found for this query.[/yellow]")
        return

    console.print(f"Fetched {len(papers)} papers. Computing embeddings...")
    embedding_model = EmbeddingModel()
    embedding_model.embed_papers(papers)

    added = store.add_many(papers)
    store.save()
    console.print(f"[green]Added {added} new papers[/green] (total in store: {len(store)})")

    console.print("Building FAISS index...")
    embedded_papers: list[Paper] = [p for p in store.all() if p.embedding is not None]
    index = VectorIndex()
    index.build(
        paper_ids=[p.paper_id for p in embedded_papers],
        embeddings=np.array([p.embedding for p in embedded_papers], dtype=np.float32),
    )
    index.save(args.index, args.id_map)
    console.print(
        f"[green]Saved FAISS index[/green] -> {args.index} ({len(embedded_papers)} vectors)"
    )


def cmd_search(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    if len(store) == 0:
        console.print("[red]No papers in store. Run `ingest` first.[/red]")
        raise SystemExit(1)

    bm25_index = BM25Index()
    bm25_index.build(store.all())

    semantic_searcher = None
    if args.mode in ("semantic", "hybrid"):
        vector_index = _load_vector_index(args)
        semantic_searcher = PaperSearcher(store=store, index=vector_index)

    hybrid_searcher = HybridSearcher(store=store, bm25_index=bm25_index, searcher=semantic_searcher)

    reranker = None
    if args.rerank:
        try:
            reranker = CrossEncoderReranker()
        except RerankerLoadError as exc:
            console.print(f"[red]Failed to load reranker:[/red] {exc}")
            raise SystemExit(1) from exc

    try:
        results = run_search(
            hybrid_searcher,
            args.query,
            mode=args.mode,
            top_k=args.top_k,
            alpha=args.alpha,
            candidate_k=args.candidate_k,
            rerank=args.rerank,
            reranker=reranker,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    if args.rerank:
        console.print(
            f"[dim]Reranked {len(results)} result(s) from up to "
            f"{max(args.candidate_k, args.top_k)} first-stage candidates[/dim]"
        )

    _print_results(results, mode=args.mode, reranked=args.rerank)


def _print_results(
    results: list[SearchResult], mode: str = "semantic", reranked: bool = False
) -> None:
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    show_breakdown = mode == "hybrid"
    title = f"Search Results ({mode}{' + reranked' if reranked else ''})"

    table = Table(title=title)
    table.add_column("Rank", justify="right")
    if show_breakdown:
        table.add_column("Semantic", justify="right")
        table.add_column("Keyword", justify="right")
        table.add_column("Retrieval", justify="right")
    else:
        table.add_column("Retrieval", justify="right")
    if reranked:
        table.add_column("Rerank", justify="right")
    table.add_column("arXiv ID")
    table.add_column("Title")
    table.add_column("Published")

    for r in results:
        d = r.to_display_dict()
        row = [str(d["rank"])]
        if show_breakdown:
            row += [d.get("semantic_score", "-"), d.get("keyword_score", "-"), d["score"]]
        else:
            row += [d["score"]]
        if reranked:
            row += [d.get("rerank_score", "-")]
        row += [d["arxiv_id"], d["title"], d["published"]]
        table.add_row(*row)
    console.print(table)

    for r in results:
        line = f"\n[bold]{r.rank}. {r.paper.title}[/bold]  (retrieval: {r.score:.3f}"
        if r.rerank_score is not None:
            line += f", rerank: {r.rerank_score:.3f}"
        line += ")"
        console.print(line)
        console.print(f"   Authors: {', '.join(r.paper.authors) or 'n/a'}")
        console.print(f"   {r.paper.abstract_url}")
        console.print(f"   {r.short_abstract()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper_explorer", description="Intelligent Research Paper Explorer (Version 3)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--store", default=str(DEFAULT_STORE_PATH), help="Path to the paper metadata store (JSON)"
    )
    parser.add_argument(
        "--index", default=str(DEFAULT_INDEX_PATH), help="Path to the FAISS index file"
    )
    parser.add_argument(
        "--id-map", default=str(DEFAULT_ID_MAP_PATH), help="Path to the FAISS id-map file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser(
        "ingest", help="Crawl arXiv, embed papers, and build the FAISS index"
    )
    p_ingest.add_argument("--query", required=True, help="Free-text arXiv search query")
    p_ingest.add_argument("--max-results", type=int, default=100)
    p_ingest.add_argument("--start", type=int, default=0, help="Offset into arXiv's result set")
    p_ingest.add_argument("--category", default=None, help="Optional arXiv category, e.g. cs.LG")
    p_ingest.add_argument(
        "--raw-dir", default=str(RAW_DIR), help="Where to save raw arXiv API responses"
    )
    p_ingest.add_argument("--no-raw", action="store_true", help="Skip saving raw API responses")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = subparsers.add_parser(
        "search", help="Search stored papers: semantic, keyword (BM25), or hybrid; optional rerank"
    )
    p_search.add_argument("query", help="Natural-language or keyword search query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument(
        "--mode",
        choices=["semantic", "keyword", "hybrid"],
        default="semantic",
        help="Retrieval mode (default: semantic, matches Version 1 behavior)",
    )
    p_search.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid weighting in [0,1]: hybrid_score = alpha*semantic + (1-alpha)*keyword "
        "(hybrid mode only)",
    )
    p_search.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Candidates retrieved per retriever before merging (hybrid mode) and/or handed "
        "to the reranker (--rerank, any mode)",
    )
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking to the first-stage candidates before returning "
        "the top --top-k results",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
