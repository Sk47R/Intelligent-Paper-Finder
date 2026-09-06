from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from paper_explorer.data.storage import PaperStore
from paper_explorer.search.engine import SearchEngine

console = Console()

MENU = """
[bold cyan]Paper Explorer[/bold cyan]
1. Semantic search
2. Recommend similar papers
3. Show a paper's abstract
4. Quit
"""


def run_ui(store_path: str) -> None:
    store = PaperStore(store_path)
    if len(store) == 0:
        console.print("[red]Store is empty. Run the `crawl` command first.[/red]")
        return

    console.print("[dim]Loading search engine (embedding model)...[/dim]")
    engine = SearchEngine(store.all())

    while True:
        console.print(MENU)
        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            query = Prompt.ask("Search query")
            top_k = IntPrompt.ask("How many results?", default=10)
            _show_results(engine.search(query, top_k=top_k))
        elif choice == "2":
            paper_id = Prompt.ask("Paper ID")
            try:
                _show_results(engine.recommend(paper_id, top_k=10))
            except KeyError as exc:
                console.print(f"[red]{exc}[/red]")
        elif choice == "3":
            paper_id = Prompt.ask("Paper ID")
            paper = store.get(paper_id)
            if paper is None:
                console.print("[red]Paper not found.[/red]")
            else:
                console.print(f"[bold]{paper.title}[/bold]\n{paper.abstract}")
        else:
            console.print("Goodbye!")
            break


def _show_results(results: list[tuple]) -> None:
    table = Table(title="Results")
    table.add_column("Score", justify="right")
    table.add_column("ID")
    table.add_column("Title")
    for paper, score in results:
        table.add_row(f"{score:.3f}", paper.paper_id, paper.title)
    console.print(table)
