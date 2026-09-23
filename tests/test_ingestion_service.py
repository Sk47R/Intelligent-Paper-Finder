import numpy as np
import pytest

from paper_explorer.data.models import Paper
from paper_explorer.data.repository import PaperRepository
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.ingestion.service import IngestionService


class FakeArxivClient:
    def __init__(self, papers):
        self._papers = papers

    def search(self, query, max_results=100, start=0, category=None, raw_dir=None):
        return self._papers


class CountingSentenceTransformer:
    def __init__(self):
        self.embedded_texts: list[str] = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.embedded_texts.extend(texts)
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def make_paper(paper_id, title="Title", abstract="Abstract"):
    return Paper(paper_id=paper_id, title=title, abstract=abstract)


@pytest.fixture
def fake_model():
    return CountingSentenceTransformer()


def test_ingest_embeds_only_new_papers(tmp_path, fake_model):
    repo = PaperRepository(tmp_path / "test.db")
    client = FakeArxivClient([make_paper("p1"), make_paper("p2")])
    embedding_model = EmbeddingModel(model=fake_model)
    service = IngestionService(repo, client=client, embedding_model=embedding_model)

    summary = service.ingest("query")
    assert summary["new"] == 2
    assert summary["embedded"] == 2
    assert len(fake_model.embedded_texts) == 2


def test_rerunning_ingest_does_not_reembed_unchanged_papers(tmp_path, fake_model):
    repo = PaperRepository(tmp_path / "test.db")
    client = FakeArxivClient([make_paper("p1"), make_paper("p2")])
    embedding_model = EmbeddingModel(model=fake_model)
    service = IngestionService(repo, client=client, embedding_model=embedding_model)

    service.ingest("query")
    assert len(fake_model.embedded_texts) == 2

    summary = service.ingest("query")
    assert summary["new"] == 0
    assert summary["unchanged"] == 2
    assert summary["embedded"] == 0
    assert len(fake_model.embedded_texts) == 2


def test_changed_paper_is_reembedded_on_next_ingest(tmp_path, fake_model):
    repo = PaperRepository(tmp_path / "test.db")
    embedding_model = EmbeddingModel(model=fake_model)

    service = IngestionService(
        repo,
        client=FakeArxivClient([make_paper("p1", abstract="Original")]),
        embedding_model=embedding_model,
    )
    service.ingest("query")
    assert len(fake_model.embedded_texts) == 1

    service.client = FakeArxivClient([make_paper("p1", abstract="Completely different text")])
    summary = service.ingest("query")
    assert summary["changed"] == 1
    assert summary["embedded"] == 1
    assert len(fake_model.embedded_texts) == 2


def test_ingestion_is_resumable_after_simulated_partial_failure(tmp_path, fake_model):
    db_path = tmp_path / "test.db"
    embedding_model = EmbeddingModel(model=fake_model)

    repo1 = PaperRepository(db_path)
    service1 = IngestionService(
        repo1,
        client=FakeArxivClient([make_paper("p1"), make_paper("p2")]),
        embedding_model=embedding_model,
    )
    service1.ingest("query")
    repo1.close()

    repo2 = PaperRepository(db_path)
    service2 = IngestionService(
        repo2,
        client=FakeArxivClient([make_paper("p1"), make_paper("p2"), make_paper("p3")]),
        embedding_model=embedding_model,
    )
    summary = service2.ingest("query")

    assert summary["new"] == 1
    assert summary["unchanged"] == 2
    assert len(fake_model.embedded_texts) == 3


def test_rebuild_index_reflects_all_embedded_papers_without_reembedding(tmp_path, fake_model):
    repo = PaperRepository(tmp_path / "test.db")
    embedding_model = EmbeddingModel(model=fake_model)
    service = IngestionService(
        repo,
        client=FakeArxivClient([make_paper("p1"), make_paper("p2")]),
        embedding_model=embedding_model,
    )
    service.ingest("query")

    index = service.rebuild_index()
    assert len(index) == 2
    assert len(fake_model.embedded_texts) == 2


def test_rebuild_index_with_no_embedded_papers_returns_empty_index(tmp_path):
    repo = PaperRepository(tmp_path / "test.db")
    service = IngestionService(repo, client=FakeArxivClient([]))
    index = service.rebuild_index()
    assert len(index) == 0
