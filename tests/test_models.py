from paper_explorer.data.models import Paper


def test_paper_to_dict_from_dict_roundtrip():
    paper = Paper(paper_id="1234.5678", title="A Title", abstract="An abstract.")
    data = paper.to_dict()
    restored = Paper.from_dict(data)
    assert restored == paper


def test_text_for_embedding_combines_title_and_abstract():
    paper = Paper(paper_id="1", title="Title", abstract="Abstract")
    text = paper.text_for_embedding
    assert "Title" in text
    assert "Abstract" in text
