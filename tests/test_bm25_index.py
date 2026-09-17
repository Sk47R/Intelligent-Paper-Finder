from paper_explorer.data.models import Paper
from paper_explorer.search.bm25_index import BM25Index, tokenize


def make_paper(paper_id: str, title: str, abstract: str) -> Paper:
    return Paper(paper_id=paper_id, title=title, abstract=abstract)


def test_tokenize_lowercases_and_strips_punctuation():
    tokens = tokenize("Transformer-based, NLP Models!")
    assert tokens == ["transformer", "based", "nlp", "models"]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []
    assert tokenize("   !!!   ") == []


def test_exact_keyword_match_ranks_highest():
    papers = [
        make_paper("p1", "Transformer Architectures for NLP", "attention mechanisms"),
        make_paper("p2", "Cooking Recipes", "how to bake bread"),
        make_paper("p3", "Graph Neural Networks", "message passing on graphs"),
    ]
    index = BM25Index()
    index.build(papers)

    results = index.search("transformer NLP", top_k=3)
    assert results[0][0] == "p1"


def test_search_returns_empty_list_when_not_built():
    index = BM25Index()
    assert index.search("anything", top_k=5) == []


def test_build_with_empty_paper_list():
    index = BM25Index()
    index.build([])
    assert not index.is_built
    assert index.search("query", top_k=5) == []


def test_top_k_larger_than_corpus_returns_all_papers():
    papers = [make_paper("p1", "Title One", "abstract one")]
    index = BM25Index()
    index.build(papers)
    results = index.search("title", top_k=100)
    assert len(results) == 1


def test_query_with_no_matching_tokens_returns_empty_or_zero_scores():
    papers = [make_paper("p1", "Graph Neural Networks", "message passing")]
    index = BM25Index()
    index.build(papers)
    results = index.search("!!!", top_k=5)
    assert results == []
