from __future__ import annotations
import numpy as np
from paper_explorer.data.models import Paper

class VectorIndex:
    def __init__(self, papers: list[Paper]):
        self.papers = [p for p in papers if p.embedding is not None]
        if self.papers:
            self._matrix = np.array([p.embedding for p in self.papers], dtype = np.float32)
        else:
            self._matrix = np.zeros((0, 0), dtype = np.float32)
            

    def __len__(self) -> int:
        return len(self.papers)
    
    def query(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[Paper, float]]:
        if len(self.papers) == 0:
            return []
        
        scores = self._matrix @ vector
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self.papers[i], float(scores[i])) for i in top_indices]
    
    def most_similar_to_paper(self, paper: Paper, top_k: int = 10) -> list[tuple[Paper, float]]:
        if paper.embedding is None:
            raise ValueError(f"Paper {paper.paper_id!r} has no embedding")
        vector = np.array(paper.embedding, dtype = np.float32)
        results = self.query(vector, top_k = top_k + 1)
        return [(p, s) for p, s in results if p.paper_id != paper.paper_id][:top_k]
    
