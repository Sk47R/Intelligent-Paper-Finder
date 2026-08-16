from __future__ import annotations

import json 
from pathlib import Path

from paper_explorer.data.models import Paper

class PaperStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._papers : dict[str, Paper] = {}
        if self.path.exists():
            self.load()
            
    def add(self, paper: Paper) -> None:
        self._papers[paper.paper_id] = paper
        
    def add_many(self, papers: list[Paper]) -> int:
        added = 0
        for paper in papers:
            if paper.paper_id not in self._papers:
                added += 1
            self.add(paper)
        return added
    
    def get(self, paper_id: str) -> Paper | None:
        return self._papers.get(paper_id)
    
    def all(self) -> list[Paper]:
        return list(self._papers.values())
    
    def __len__(self) -> int:
        return len(self._papers)
    
    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self._papers
    
    def save(self) -> None:
        self.path.parent.mkdir(parents = True, exist_ok = True)
        data = [p.to_dict() for p in self._papers.values()]
        self.path.write_text(json.dumps(data, indent = 2), encoding = "utf-8")
        
    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding = "utf-8"))
        self._papers = {d["paper_id"]: Paper.from_dict(d) for d in data}