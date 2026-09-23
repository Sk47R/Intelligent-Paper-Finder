from paper_explorer.data.database import get_connection, init_schema


def test_init_schema_creates_expected_tables(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_schema(conn)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "papers" in tables
    assert "embeddings" in tables
    conn.close()


def test_init_schema_is_idempotent(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_schema(conn)
    init_schema(conn)
    conn.close()
