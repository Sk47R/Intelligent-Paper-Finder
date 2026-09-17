from paper_explorer.cli import build_parser


def test_ingest_parses_required_and_optional_flags():
    parser = build_parser()
    args = parser.parse_args(["ingest", "--query", "transformer NLP", "--max-results", "50"])
    assert args.command == "ingest"
    assert args.query == "transformer NLP"
    assert args.max_results == 50
    assert args.start == 0
    assert args.category is None


def test_ingest_requires_query_flag():
    parser = build_parser()
    try:
        parser.parse_args(["ingest", "--max-results", "50"])
        assert False, "expected SystemExit (missing --query)"
    except SystemExit:
        pass


def test_search_parses_positional_query_and_top_k():
    parser = build_parser()
    args = parser.parse_args(["search", "transformer architectures", "--top-k", "5"])
    assert args.command == "search"
    assert args.query == "transformer architectures"
    assert args.top_k == 5


def test_search_defaults_to_semantic_mode():
    parser = build_parser()
    args = parser.parse_args(["search", "transformer architectures"])
    assert args.mode == "semantic"
    assert args.alpha == 0.5
    assert args.candidate_k == 50


def test_search_parses_hybrid_mode_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["search", "transformer in NLP", "--mode", "hybrid", "--alpha", "0.7", "--candidate-k", "50", "--top-k", "10"]
    )
    assert args.mode == "hybrid"
    assert args.alpha == 0.7
    assert args.candidate_k == 50
    assert args.top_k == 10