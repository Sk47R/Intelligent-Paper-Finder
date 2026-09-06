import numpy as np

from paper_explorer.data.models import Paper
from paper_explorer.search.index import VectorIndex


def make_paper(paper_id: str, vector: list[float]) -> Paper:
    return Paper(paper_id=paper_id, title=paper_id, abstract="", embedding=vector)


def test_query_returns_best_match_first():
    papers = [
        make_paper("a", [1.0, 0.0]),
        make_paper("b", [0.0, 1.0]),
        make_paper("c", [0.9, 0.1]),
    ]
    index = VectorIndex(papers)
    results = index.query(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    ids = [p.paper_id for p, _ in results]
    assert ids[0] == "a"
    assert "b" not in ids


def test_most_similar_excludes_self():
    papers = [
        make_paper("a", [1.0, 0.0]),
        make_paper("b", [0.9, 0.1]),
    ]
    index = VectorIndex(papers)
    results = index.most_similar_to_paper(papers[0], top_k=5)
    ids = [p.paper_id for p, _ in results]
    assert "a" not in ids
    assert "b" in ids
