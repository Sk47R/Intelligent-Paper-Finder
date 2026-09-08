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
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.results import SearchResult
from paper_explorer.search.searcher import PaperSearcher

console = Console()
logger = logging.getLogger("paper_explorer")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level = logging.DEBUG if verbose else logging.INFO,
        format = "%(message)s",
        handlers = [RichHandler(console=console, show_path=False)],
        force = True,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    client = ArxivClient()

    console.print(f"[bold]Querying arXiv for[/bold] {args.query!r} ...")
    try:
        papers = client.search(
            args.query,
            max_results = args.max_results,
            start = args.start,
            category = args.category,
            raw_dir = None if args.no_raw else args.raw_dir,
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

    try:
        index = VectorIndex.load(args.index, args.id_map)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(f"[red]No FAISS index found at {args.index}. Run `ingest` first.[/red]")
        raise SystemExit(1) from exc

    searcher = PaperSearcher(store=store, index=index)
    try:
        results = searcher.search(args.query, top_k=args.top_k)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    _print_results(results)


def _print_results(results: list[SearchResult]) -> None:
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("Rank", justify = "right")
    table.add_column("Score", justify = "right")
    table.add_column("arXiv ID")
    table.add_column("Title")
    table.add_column("Published")
    for r in results:
        d = r.to_display_dict()
        table.add_row(str(d["rank"]), d["score"], d["arxiv_id"], d["title"], d["published"])
    console.print(table)

    for r in results:
        console.print(f"\n[bold]{r.rank}. {r.paper.title}[/bold]  (score: {r.score:.3f})")
        console.print(f"   Authors: {', '.join(r.paper.authors) or 'n/a'}")
        console.print(f"   {r.paper.abstract_url}")
        console.print(f"   {r.short_abstract()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "paper_explorer", description = "Intelligent Research Paper Explorer (Version 1)"
    )
    parser.add_argument("-v", "--verbose", action = "store_true", help = "Enable debug logging")
    parser.add_argument(
        "--store", default = str(DEFAULT_STORE_PATH), help = "Path to the paper metadata store (JSON)"
    )
    parser.add_argument(
        "--index", default = str(DEFAULT_INDEX_PATH), help = "Path to the FAISS index file"
    )
    parser.add_argument(
        "--id-map", default = str(DEFAULT_ID_MAP_PATH), help = "Path to the FAISS id-map file"
    )
    subparsers = parser.add_subparsers(dest = "command", required = True)

    p_ingest = subparsers.add_parser(
        "ingest", help = "Crawl arXiv, embed papers, and build the FAISS index"
    )
    p_ingest.add_argument("--query", required=True, help = "Free-text arXiv search query")
    p_ingest.add_argument("--max-results", type = int, default = 100)
    p_ingest.add_argument("--start", type = int, default = 0, help = "Offset into arXiv's result set")
    p_ingest.add_argument("--category", default = None, help = "Optional arXiv category, e.g. cs.LG")
    p_ingest.add_argument(
        "--raw-dir", default = str(RAW_DIR), help = "Where to save raw arXiv API responses"
    )
    p_ingest.add_argument(
        "--no-raw", action = "store_true", help = "Skip saving raw API responses"
    )
    p_ingest.set_defaults(func = cmd_ingest)

    p_search = subparsers.add_parser("search", help = "Semantic search over the FAISS index")
    p_search.add_argument("query", help = "Natural-language search query")
    p_search.add_argument("--top-k", type = int, default = 10)
    p_search.set_defaults(func = cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())