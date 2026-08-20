from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from paper_explorer.data.models import Paper

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts, 
            show_progress_bar = False,
            convert_to_numpy = True, 
            normalize_embeddings = True,
        )
        return embeddings.astype(np.float32)
    
    def embed_papers(self, papers: list[Paper]) -> list[Paper]:
        if not papers:
            return papers
        
        texts = [paper.text_for_embedding for paper in papers]
        vectors = self.embed_texts(texts)
        
        for paper, vector in zip(papers, vectors):
            paper.embedding = vector.tolist()
            
        return papers