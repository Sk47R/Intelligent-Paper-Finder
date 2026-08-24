from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from paper_explorer.data.models import Paper


@dataclass
class Topic:
    topic_id: int
    keywords: list[str]
    papers: list[Paper]
    
    def summary(self) -> str:
        keyword_str = ", ".join(self.keywords)
        example_title = self.papers[0].title if self.papers else "n/a"
        return (
            f"Topic {self.topic_id}: {keyword_str} "
            f"({len(self.papers)} papers, e.g. {example_title!r})"
        )
        
class TopicSummarizer:
    def __init__(self, n_topics: int = 5, n_keywords: int = 8, random_state: int = 42):
        self.n_topics = n_topics
        self.n_keywords = n_keywords
        self.random_state = random_state
        
    def fit(self, papers: list[Paper]) -> list[Topic]:
        embedded = [p for p in papers if p.embedding is not None]
        if len(embedded) < self.n_topics:
            raise ValueError(
                f"Need at least {self.n_topics} embedded papers, got {len(embedded)}"
            )
        
        matrix = np.array([p.embedding for p in embedded], dtype = np.float32)
        kmeans = KMeans(n_clusters = self.n_topics, random_state = self.random_state, n_init = 10)
        labels = kmeans.fit_predict(matrix)
        
        topics = []
        for topic_id in range(self.n_topics):
            cluster_papers = [p for p, label in zip(embedded, labels) if label == topic_id]
            if not cluster_papers:
                continue
            keywords = self._top_keywords(cluster_papers)
            topics.append(Topic(topic_id = topic_id, keywords = keywords, papers = cluster_papers))
            
        return topics
    
    
    def _top_keywords(self, papers: list[Paper]) -> list[str]:
        texts = [p.text_for_embedding for p in papers]
        vectorizer = TfidfVectorizer(stop_words = "english", max_features = 2000, ngram_range = (1, 2))
        tfidf = vectorizer.fit_transform(texts)
        scores = np.asarray(tfidf.sum(axis = 0)).ravel()
        topic_indices = np.argsort(-scores)[: self.n_keywords]
        vocab = vectorizer.get_feature_names_out()
        return [vocab[i] for i in topic_indices]