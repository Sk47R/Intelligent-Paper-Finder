import numpy as np

from paper_explorer.data.models import Paper
from paper_explorer.data.repository import PaperRepository
from paper_explorer.data.storage import PaperStore


def test_migrate_json_papers_into_repository(tmp_path):
    json_store = PaperStore(tmp_path / "papers.json")
    p1 = Paper(paper_id="p1", title="T1", abstract="A1", embedding=[0.1, 0.2])
    p2 = Paper(paper_id="p2", title="T2", abstract="A2", embedding=None)
    json_store.add_many([p1, p2])
    json_store.save()

    reloaded = PaperStore(tmp_path / "papers.json")
    repo = PaperRepository(tmp_path / "papers.db")
    for paper in reloaded.all():
        repo.upsert_paper(paper)
        if paper.embedding is not None:
            repo.set_embedding(
                paper.paper_id, np.array(paper.embedding, dtype=np.float32), "migrated-model"
            )

    assert len(repo) == 2
    pairs = repo.get_all_embedded_vectors()
    assert len(pairs) == 1
    assert pairs[0][0] == "p1"
