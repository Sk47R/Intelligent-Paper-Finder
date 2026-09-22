from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from paper_explorer.data.content_hash import compute_content_hash
from paper_explorer.data.database import get_connection, init_schema
from paper_explorer.data.models import Paper

EMBEDDED = "embedded"
PENDING = "pending"
STALE = "stale"
FAILED = "failed"


class PaperRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection = get_connection(self.db_path)
        init_schema(self.conn)

    def close(self) -> None:
        self.conn.close()

    def upsert_paper(self, paper: Paper) -> str:
        new_hash = compute_content_hash(paper.title, paper.abstract)
        row = self.conn.execute(
            "SELECT content_hash FROM papers WHERE paper_id = ?", (paper.paper_id,)
        ).fetchone()

        authors_json = json.dumps(paper.authors)
        categories_json = json.dumps(paper.categories)

        if row is None:
            self.conn.execute(
                """
                INSERT INTO papers (
                    paper_id, title, abstract, authors, categories,
                    published, updated, pdf_url, abstract_url, source,
                    content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.paper_id,
                    paper.title,
                    paper.abstract,
                    authors_json,
                    categories_json,
                    paper.published,
                    paper.updated,
                    paper.pdf_url,
                    paper.abstract_url,
                    paper.source,
                    new_hash,
                ),
            )
            self.conn.execute(
                "INSERT INTO embeddings (paper_id, status) VALUES (?, ?)",
                (paper.paper_id, PENDING),
            )
            self.conn.commit()
            return "new"

        old_hash = row["content_hash"]
        self.conn.execute(
            """
            UPDATE papers SET title=?, abstract=?, authors=?, categories=?,
                published=?, updated=?, pdf_url=?, abstract_url=?, source=?,
                content_hash=?, updated_at=datetime('now')
            WHERE paper_id=?
            """,
            (
                paper.title,
                paper.abstract,
                authors_json,
                categories_json,
                paper.published,
                paper.updated,
                paper.pdf_url,
                paper.abstract_url,
                paper.source,
                new_hash,
                paper.paper_id,
            ),
        )
        if new_hash != old_hash:
            self.conn.execute(
                "UPDATE embeddings SET status=? WHERE paper_id=?", (STALE, paper.paper_id)
            )
            self.conn.commit()
            return "changed"

        self.conn.commit()
        return "unchanged"

    def set_embedding(self, paper_id: str, vector: np.ndarray, model_name: str) -> None:
        row = self.conn.execute(
            "SELECT content_hash FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Paper {paper_id!r} not found; cannot set embedding")

        vector = np.asarray(vector, dtype=np.float32)
        self.conn.execute(
            """
            UPDATE embeddings
            SET model_name=?, dimension=?, content_hash=?, status=?,
                vector=?, updated_at=datetime('now')
            WHERE paper_id=?
            """,
            (
                model_name,
                vector.shape[0],
                row["content_hash"],
                EMBEDDED,
                vector.tobytes(),
                paper_id,
            ),
        )
        self.conn.commit()

    def mark_embedding_failed(self, paper_id: str) -> None:
        self.conn.execute(
            "UPDATE embeddings SET status=?, updated_at=datetime('now') WHERE paper_id=?",
            (FAILED, paper_id),
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM papers")
        self.conn.commit()

    def get(self, paper_id: str) -> Paper | None:
        row = self.conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return self._row_to_paper(row) if row else None

    def all(
        self,
        category: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[Paper]:
        query = "SELECT * FROM papers WHERE 1=1"
        params: list = []
        if from_date:
            query += " AND published >= ?"
            params.append(from_date)
        if to_date:
            query += " AND published <= ?"
            params.append(to_date)
        rows = self.conn.execute(query, params).fetchall()
        papers = [self._row_to_paper(r) for r in rows]
        if category:
            papers = [p for p in papers if category in p.categories]
        return papers

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def __contains__(self, paper_id: str) -> bool:
        return self.get(paper_id) is not None

    def get_papers_needing_embedding(self) -> list[Paper]:
        rows = self.conn.execute(
            """
            SELECT p.* FROM papers p
            JOIN embeddings e ON e.paper_id = p.paper_id
            WHERE e.status IN (?, ?)
            """,
            (PENDING, STALE),
        ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_all_embedded_vectors(self) -> list[tuple[str, np.ndarray]]:
        rows = self.conn.execute(
            "SELECT paper_id, dimension, vector FROM embeddings "
            "WHERE status=? AND vector IS NOT NULL",
            (EMBEDDED,),
        ).fetchall()
        results = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dimension"])
            results.append((row["paper_id"], vector))
        return results

    def stats(self) -> dict:
        total_papers = len(self)
        embedded = self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status=?", (EMBEDDED,)
        ).fetchone()[0]
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status=?", (PENDING,)
        ).fetchone()[0]
        stale = self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status=?", (STALE,)
        ).fetchone()[0]
        failed = self.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status=?", (FAILED,)
        ).fetchone()[0]
        model_row = self.conn.execute(
            "SELECT DISTINCT model_name, dimension FROM embeddings WHERE status=? LIMIT 1",
            (EMBEDDED,),
        ).fetchone()
        date_row = self.conn.execute(
            "SELECT MIN(published), MAX(published) FROM papers WHERE published IS NOT NULL"
        ).fetchone()
        category_rows = self.conn.execute("SELECT categories FROM papers").fetchall()
        categories: set[str] = set()
        for r in category_rows:
            categories.update(json.loads(r["categories"]))

        return {
            "total_papers": total_papers,
            "embedded": embedded,
            "pending": pending,
            "stale": stale,
            "failed": failed,
            "embedding_model": model_row["model_name"] if model_row else None,
            "embedding_dimension": model_row["dimension"] if model_row else None,
            "categories": sorted(categories),
            "date_range": (date_row[0], date_row[1]) if date_row else (None, None),
        }

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors"]),
            categories=json.loads(row["categories"]),
            published=row["published"],
            updated=row["updated"],
            pdf_url=row["pdf_url"],
            abstract_url=row["abstract_url"],
            source=row["source"],
        )
