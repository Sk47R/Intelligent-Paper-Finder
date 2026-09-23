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
        [
            "search",
            "transformer in NLP",
            "--mode",
            "hybrid",
            "--alpha",
            "0.7",
            "--candidate-k",
            "50",
            "--top-k",
            "10",
        ]
    )
    assert args.mode == "hybrid"
    assert args.alpha == 0.7
    assert args.candidate_k == 50
    assert args.top_k == 10


def test_search_rerank_flag_defaults_to_false():
    parser = build_parser()
    args = parser.parse_args(["search", "transformer architectures"])
    assert args.rerank is False


def test_search_parses_rerank_flag_with_hybrid_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "search",
            "transformer architectures for NLP",
            "--mode",
            "hybrid",
            "--candidate-k",
            "50",
            "--top-k",
            "10",
            "--rerank",
        ]
    )
    assert args.rerank is True
    assert args.mode == "hybrid"
    assert args.candidate_k == 50
    assert args.top_k == 10


def test_ingest_has_db_flag_with_default():
    parser = build_parser()
    args = parser.parse_args(["ingest", "--query", "transformer NLP"])
    assert args.db


def test_migrate_parses_from_json_flag():
    parser = build_parser()
    args = parser.parse_args(["migrate", "--from-json", "data/processed/papers.json"])
    assert args.command == "migrate"
    assert args.from_json == "data/processed/papers.json"


def test_index_rebuild_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["index", "rebuild"])
    assert args.command == "index"
    assert args.index_command == "rebuild"


def test_stats_command_parses():
    parser = build_parser()
    args = parser.parse_args(["stats"])
    assert args.command == "stats"


def test_reset_requires_explicit_yes_flag():
    parser = build_parser()
    assert parser.parse_args(["reset"]).yes is False
    assert parser.parse_args(["reset", "--yes"]).yes is True


def test_search_parses_category_and_date_filters():
    parser = build_parser()
    args = parser.parse_args(
        [
            "search",
            "transformer NLP",
            "--category",
            "cs.CL",
            "--from-date",
            "2022-01-01",
            "--to-date",
            "2023-01-01",
            "--top-k",
            "10",
        ]
    )
    assert args.category == "cs.CL"
    assert args.from_date == "2022-01-01"
    assert args.to_date == "2023-01-01"
