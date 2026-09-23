from paper_explorer.data.content_hash import compute_content_hash


def test_same_title_and_abstract_produce_same_hash():
    assert compute_content_hash("Title", "Abstract text") == compute_content_hash(
        "Title", "Abstract text"
    )


def test_different_abstract_produces_different_hash():
    assert compute_content_hash("Title", "Abstract one") != compute_content_hash(
        "Title", "Abstract two"
    )


def test_hash_ignores_surrounding_whitespace():
    assert compute_content_hash("Title", "Abstract") == compute_content_hash(
        "  Title  ", "  Abstract  "
    )
