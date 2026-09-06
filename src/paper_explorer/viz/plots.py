from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from paper_explorer.data.models import Paper


def plot_papers_per_year(papers: list[Paper], output_path: str | Path) -> Path:
    years = [p.published[:4] for p in papers if p.published]
    counts = Counter(years)
    ordered = sorted(counts.items())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([year for year, _ in ordered], [count for _, count in ordered], color="#3366cc")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Papers per year")
    plt.xticks(rotation=45)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_embedding_space(
    papers: list[Paper], output_path: str | Path, labels: list[int] | None = None
) -> Path:
    embedded = [p for p in papers if p.embedding is not None]
    matrix = np.array([p.embedding for p in embedded], dtype=np.float32)
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    ax.set_title("Paper embedding space (PCA projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if labels is not None:
        legend = ax.legend(*scatter.legend_elements(), title="Topic", loc="best")
        ax.add_artist(legend)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
