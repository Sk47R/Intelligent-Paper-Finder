from pathlib import Path

from paper_explorer.data.models import Paper
from paper_explorer.data.storage import PaperStore


def make_paper(paper_id: str) -> Paper:
    return Paper(paper_id=paper_id, title=f"Title {paper_id}", abstract="Abstract text")


def test_add_and_get(tmp_path: Path):
    store = PaperStore(tmp_path / "papers.json")
    store.add(make_paper("p1"))
    assert len(store) == 1
    assert store.get("p1").title == "Title p1"


def test_add_many_counts_only_new(tmp_path: Path):
    store = PaperStore(tmp_path / "papers.json")
    added_first = store.add_many([make_paper("p1"), make_paper("p2")])
    added_second = store.add_many([make_paper("p2"), make_paper("p3")])
    assert added_first == 2
    assert added_second == 1
    assert len(store) == 3


def test_save_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "papers.json"
    store = PaperStore(path)
    store.add(make_paper("p1"))
    store.save()

    reloaded = PaperStore(path)
    assert len(reloaded) == 1
    assert reloaded.get("p1").title == "Title p1"