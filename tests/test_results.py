from paper_explorer.data.models import Paper
from paper_explorer.search.results import SearchResult


def make_paper(abstract_len: int = 300) -> Paper:
    return Paper(
        paper_id = "1234.5678",
        title = "A Paper",
        abstract = "x" * abstract_len,
        authors = ["A. One", "B. Two"],
        abstract_url = "https://arxiv.org/abs/1234.5678",
    )


def test_short_abstract_truncates_long_text():
    result = SearchResult(rank = 1, paper = make_paper(300), score = 0.9)
    short = result.short_abstract(max_chars = 50)
    assert len(short) <= 53
    assert short.endswith("...")


def test_short_abstract_keeps_short_text_unchanged():
    paper = make_paper(abstract_len = 10)
    result = SearchResult(rank = 1, paper = paper, score = 0.9)
    assert result.short_abstract(max_chars = 50) == paper.abstract


def test_to_display_dict_has_expected_keys():
    result = SearchResult(rank = 1, paper = make_paper(), score = 0.876)
    d = result.to_display_dict()
    assert d["rank"] == 1
    assert d["score"] == "0.876"
    assert d["authors"] == "A. One, B. Two"
    assert d["url"] == "https://arxiv.org/abs/1234.5678"