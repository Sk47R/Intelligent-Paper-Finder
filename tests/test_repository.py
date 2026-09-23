import numpy as np
import pytest

from paper_explorer.data.models import Paper
from paper_explorer.data.repository import PaperRepository


def make_paper(paper_id="p1", title="A Title", abstract="An abstract", **kwargs):
    return Paper(paper_id=paper_id, title=title, abstract=abstract, **kwargs)


@pytest.fixture
def repo(tmp_path):
    return PaperRepository(tmp_path / "test.db")


def test_upsert_new_paper_returns_new(repo):
    assert repo.upsert_paper(make_paper()) == "new"
    assert len(repo) == 1


def test_upsert_existing_unchanged_returns_unchanged_no_duplicate(repo):
    repo.upsert_paper(make_paper())
    assert repo.upsert_paper(make_paper()) == "unchanged"
    assert len(repo) == 1


def test_upsert_changed_abstract_returns_changed(repo):
    repo.upsert_paper(make_paper(abstract="Original abstract"))
    status = repo.upsert_paper(make_paper(abstract="A completely different abstract"))
    assert status == "changed"
    assert len(repo) == 1


def test_metadata_only_change_returns_unchanged_but_still_updates_fields(repo):
    repo.upsert_paper(make_paper(categories=["cs.LG"]))
    status = repo.upsert_paper(make_paper(categories=["cs.LG", "cs.CL"]))
    assert status == "unchanged"
    assert repo.get("p1").categories == ["cs.LG", "cs.CL"]


def test_get_returns_none_for_missing_paper(repo):
    assert repo.get("does-not-exist") is None


def test_new_paper_is_pending_embedding(repo):
    repo.upsert_paper(make_paper())
    pending = repo.get_papers_needing_embedding()
    assert [p.paper_id for p in pending] == ["p1"]


def test_set_embedding_removes_paper_from_pending(repo):
    repo.upsert_paper(make_paper())
    repo.set_embedding("p1", np.array([0.1, 0.2, 0.3], dtype=np.float32), "fake-model")
    assert repo.get_papers_needing_embedding() == []


def test_changed_abstract_marks_embedding_stale_not_recomputed(repo):
    repo.upsert_paper(make_paper(abstract="Original"))
    repo.set_embedding("p1", np.array([1.0, 0.0], dtype=np.float32), "fake-model")
    assert repo.get_papers_needing_embedding() == []

    repo.upsert_paper(make_paper(abstract="Changed abstract entirely"))
    pending = repo.get_papers_needing_embedding()
    assert [p.paper_id for p in pending] == ["p1"]


def test_unchanged_abstract_does_not_mark_embedding_stale(repo):
    repo.upsert_paper(make_paper())
    repo.set_embedding("p1", np.array([1.0, 0.0], dtype=np.float32), "fake-model")
    repo.upsert_paper(make_paper())
    assert repo.get_papers_needing_embedding() == []


def test_get_all_embedded_vectors_roundtrips_exactly(repo):
    repo.upsert_paper(make_paper())
    original = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    repo.set_embedding("p1", original, "fake-model")

    pairs = repo.get_all_embedded_vectors()
    assert len(pairs) == 1
    paper_id, vector = pairs[0]
    assert paper_id == "p1"
    assert np.allclose(vector, original)


def test_set_embedding_raises_for_unknown_paper(repo):
    with pytest.raises(KeyError):
        repo.set_embedding("nope", np.array([1.0]), "fake-model")


def test_filter_by_category(repo):
    repo.upsert_paper(make_paper(paper_id="p1", categories=["cs.LG"]))
    repo.upsert_paper(make_paper(paper_id="p2", categories=["cs.CL"]))
    assert [p.paper_id for p in repo.all(category="cs.CL")] == ["p2"]


def test_filter_by_date_range(repo):
    repo.upsert_paper(make_paper(paper_id="p1", published="2020-01-01"))
    repo.upsert_paper(make_paper(paper_id="p2", published="2023-06-01"))
    results = repo.all(from_date="2022-01-01", to_date="2024-01-01")
    assert [p.paper_id for p in results] == ["p2"]


def test_stats_reports_counts_and_model_info(repo):
    repo.upsert_paper(make_paper(paper_id="p1", categories=["cs.LG"], published="2023-01-01"))
    repo.upsert_paper(make_paper(paper_id="p2", categories=["cs.CL"], published="2023-06-01"))
    repo.set_embedding("p1", np.array([1.0, 0.0], dtype=np.float32), "fake-model")

    s = repo.stats()
    assert s["total_papers"] == 2
    assert s["embedded"] == 1
    assert s["pending"] == 1
    assert s["embedding_model"] == "fake-model"
    assert s["embedding_dimension"] == 2
    assert set(s["categories"]) == {"cs.LG", "cs.CL"}
    assert s["date_range"] == ("2023-01-01", "2023-06-01")


def test_reset_clears_all_data(repo):
    repo.upsert_paper(make_paper())
    repo.reset()
    assert len(repo) == 0
    assert repo.get_papers_needing_embedding() == []


def test_data_survives_closing_and_reopening_connection(tmp_path):
    """Simulates process restart -- proves durability/resumability."""
    db_path = tmp_path / "test.db"
    repo1 = PaperRepository(db_path)
    repo1.upsert_paper(make_paper())
    repo1.close()

    repo2 = PaperRepository(db_path)
    assert len(repo2) == 1
    assert repo2.get("p1") is not None
