from __future__ import annotations
from paper_explorer.data.models import Paper
from paper_explorer.embeddings.embedder import Embedder
from paper_explorer.search.index import VectorIndex

class SearchEngine:
    def __init__(self, papers: list[Paper], embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index = VectorIndex(papers)
        
    def search(self, query:str, top_k: int = 10) -> list[tuple[Paper, float]]:
        vector = self.embedder.embed_texts([query])[0]
        return self.index.query(vector, top_k = top_k)
    
    def recommend(self, paper_id: str, top_k: int = 10) -> list[tuple[Paper, float]]:
        paper = next((p for p in self.index.papers if p.paper_id == paper_id), None)
        if paper is None:
            raise KeyError(f"Paper {paper_id!r} not found in index")
        
        return self.index.most_similar_to_paper(paper, top_k = top_k)