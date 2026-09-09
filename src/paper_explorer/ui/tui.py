from __future__ import annotations
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.searcher import PaperSearcher

console = Console()

MENU = """
[bold cyan]Paper Explorer[/bold cyan]
1. Semantic search
2. Show a paper's abstract
3. Quit
"""


def run_ui(store_path: str, index_path: str, id_map_path: str) -> None:
    store = PaperStore(store_path)
    if len(store) == 0:
        console.print("[red]Store is empty. Run `ingest` first.[/red]")
        return

    index = VectorIndex.load(index_path, id_map_path)
    searcher = PaperSearcher(store = store, index = index, embedding_model = EmbeddingModel())

    while True:
        console.print(MENU)
        choice = Prompt.ask("Choose an option", choices = ["1", "2", "3"], default = "1")

        if choice == "1":
            query = Prompt.ask("Search query")
            top_k = IntPrompt.ask("How many results?", default = 10)
            _show_results(searcher.search(query, top_k = top_k))
        elif choice == "2":
            paper_id = Prompt.ask("Paper ID")
            paper = store.get(paper_id)
            if paper is None:
                console.print("[red]Paper not found.[/red]")
            else:
                console.print(f"[bold]{paper.title}[/bold]\n{paper.abstract}")
        else:
            console.print("Goodbye!")
            break


def _show_results(results) -> None:
    table = Table(title = "Results")
    table.add_column("Rank", justify = "right")
    table.add_column("Score", justify = "right")
    table.add_column("ID")
    table.add_column("Title")
    for r in results:
        table.add_row(str(r.rank), f"{r.score:.3f}", r.paper.paper_id, r.paper.title)
    console.print(table)