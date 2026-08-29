from __future__ import annotations
import argparse
import sys
from rich.console import Console
from rich.table import Table
from paper_explorer.config import DEFAULT_PLOTS_DIR, DEFAULT_STORE_PATH
from paper_explorer.crawler.arxiv_crawler import ArxivCrawler
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedder import Embedder
from paper_explorer.search.engine import SearchEngine
from paper_explorer.summarize.topic_summary import TopicSummarizer
from paper_explorer.viz.plots import plot_embedding_space, plot_papers_per_year

console = Console()


def cmd_crawl(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    crawler = ArxivCrawler()
    console.print(f"[bold]Searching arXiv for[/bold] {args.query!r} ...")
    papers = crawler.search(args.query, max_results=args.max_results)
    console.print(f"Fetched {len(papers)} papers. Computing embeddings...")
    embedder = Embedder()
    embedder.embed_papers(papers)
    added = store.add_many(papers)
    store.save()
    console.print(
        f"[green]Added {added} new papers[/green] "
        f"(total in store: {len(store)}) -> {store.path}"
    )


def cmd_search(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    if len(store) == 0:
        console.print("[red]No papers in store. Run `crawl` first.[/red]")
        return
    engine = SearchEngine(store.all())
    results = engine.search(args.query, top_k=args.top_k)
    _print_results(results)


def cmd_recommend(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    if len(store) == 0:
        console.print("[red]No papers in store. Run `crawl` first.[/red]")
        return
    engine = SearchEngine(store.all())
    try:
        results = engine.recommend(args.paper_id, top_k=args.top_k)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    _print_results(results)


def cmd_topics(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    if len(store) == 0:
        console.print("[red]No papers in store. Run `crawl` first.[/red]")
        return
    summarizer = TopicSummarizer(n_topics=args.n_topics)
    topics = summarizer.fit(store.all())
    for topic in topics:
        console.print(topic.summary())


def cmd_plot(args: argparse.Namespace) -> None:
    store = PaperStore(args.store)
    if len(store) == 0:
        console.print("[red]No papers in store. Run `crawl` first.[/red]")
        return
    papers = store.all()
    year_plot = plot_papers_per_year(papers, DEFAULT_PLOTS_DIR / "papers_per_year.png")
    console.print(f"Saved {year_plot}")
    summarizer = TopicSummarizer(n_topics=args.n_topics)
    topics = summarizer.fit(papers)
    label_map = {p.paper_id: t.topic_id for t in topics for p in t.papers}
    embedded = [p for p in papers if p.paper_id in label_map]
    labels = [label_map[p.paper_id] for p in embedded]
    embed_plot = plot_embedding_space(embedded, DEFAULT_PLOTS_DIR / "embedding_space.png", labels)
    console.print(f"Saved {embed_plot}")


def _print_results(results: list[tuple]) -> None:
    table = Table(title="Results")
    table.add_column("Score", justify="right")
    table.add_column("ID")
    table.add_column("Title")
    for paper, score in results:
        table.add_row(f"{score:.3f}", paper.paper_id, paper.title)
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper_explorer", description="Intelligent Research Paper Explorer"
    )
    parser.add_argument(
        "--store", default=str(DEFAULT_STORE_PATH), help="Path to the local paper store (JSON)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_crawl = subparsers.add_parser("crawl", help="Crawl papers from arXiv into the store")
    p_crawl.add_argument("query", help="Free-text search query")
    p_crawl.add_argument("--max-results", type=int, default=100)
    p_crawl.set_defaults(func=cmd_crawl)

    p_search = subparsers.add_parser("search", help="Semantic search over stored papers")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_recommend = subparsers.add_parser(
        "recommend", help="Recommend papers similar to a given paper"
    )
    p_recommend.add_argument("paper_id")
    p_recommend.add_argument("--top-k", type=int, default=10)
    p_recommend.set_defaults(func=cmd_recommend)

    p_topics = subparsers.add_parser("topics", help="Discover and summarize topics in the store")
    p_topics.add_argument("--n-topics", type=int, default=5)
    p_topics.set_defaults(func=cmd_topics)

    p_plot = subparsers.add_parser("plot", help="Generate visualizations (saved as PNG files)")
    p_plot.add_argument("--n-topics", type=int, default=5)
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())