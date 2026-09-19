import pytest

from paper_explorer.data.models import Paper
from paper_explorer.search.reranker import CrossEncoderReranker, RerankerLoadError
from paper_explorer.search.results import SearchResult


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=16):
        return [1.0 if "graph" in doc.lower() else 0.0 for _, doc in pairs]


def make_result(paper_id, title, abstract, rank=1, score=0.5):
    paper = Paper(paper_id=paper_id, title=title, abstract=abstract)
    return SearchResult(rank=rank, paper=paper, score=score)


def test_rerank_reorders_by_cross_encoder_score():
    candidates = [
        make_result("p1", "Cooking Recipes", "how to bake bread", rank=1, score=0.9),
        make_result("p2", "Graph Neural Networks", "message passing on graphs", rank=2, score=0.1),
    ]
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = reranker.rerank("graph learning", candidates)

    assert results[0].paper.paper_id == "p2"
    assert results[0].rank == 1
    assert results[0].rerank_score == pytest.approx(1.0)
    assert results[1].rerank_score == pytest.approx(0.0)


def test_rerank_preserves_retrieval_score_distinct_from_rerank_score():
    candidates = [make_result("p1", "Graph Neural Networks", "graphs", score=0.42)]
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = reranker.rerank("graph", candidates)
    assert results[0].score == pytest.approx(0.42)
    assert results[0].rerank_score == pytest.approx(1.0)


def test_rerank_empty_candidates_returns_empty_list():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    assert reranker.rerank("anything", []) == []


def test_rerank_reassigns_ranks_starting_at_one():
    candidates = [
        make_result("p1", "A", "no match here", rank=5, score=0.9),
        make_result("p2", "Graph Paper", "graph content", rank=9, score=0.1),
        make_result("p3", "B", "also no match", rank=1, score=0.5),
    ]
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = reranker.rerank("graph", candidates)
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].paper.paper_id == "p2"


def test_reranker_load_error_wraps_underlying_exception(monkeypatch):
    class BrokenCrossEncoder:
        def __init__(self, model_name):
            raise RuntimeError("simulated load failure")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", BrokenCrossEncoder)

    with pytest.raises(RerankerLoadError):
        CrossEncoderReranker(model_name="fake-model-name")
