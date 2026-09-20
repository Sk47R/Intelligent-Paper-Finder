from __future__ import annotations

import hashlib


def compute_content_hash(title: str, abstract: str) -> str:
    normalized = f"{title.strip()}\n\n{abstract.strip()}".encode()
    return hashlib.sha256(normalized).hexdigest()
