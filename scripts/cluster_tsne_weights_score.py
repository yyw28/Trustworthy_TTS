#!/usr/bin/env python3
"""
Cluster and t-SNE visualization using GST weights and trustworthiness score from scores.json.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_scores(scores_path: Path) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load scores.json; return (weights_matrix, scores_array, records)."""
    data = json.loads(scores_path.read_text())
    weights = np.array([r["gst_weights"] for r in data], dtype=np.float64)
    scores = np.array([r["trustworthiness_score"] for r in data], dtype=np.float64)
    return weights, scores, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster and t-SNE on GST weights + score")
    parser.add_argument(
        "scores_json",
        type=Path,
        default=Path("generated_audio_test_1000/scores.json"),
        nargs="?",
        help="Path to scores.json",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=2,
        help="Number of K-means clusters",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (5–50 typical)",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="Random state for t-SNE",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: same as scores.json parent)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use only GST weights (10 dims) for t-SNE and clustering; still color by score in second plot.",
    )
    args = parser.parse_args()

    scores_path = args.scores_json.resolve()
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    weights, scores, records = load_scores(scores_path)
    n = len(records)

    if args.weights_only:
        X = weights.copy()
    else:
        X = np.hstack([weights, scores.reshape(-1, 1)])
    valid = np.isfinite(X).all(axis=1)
    if not valid.all():
        X = X[valid]
        scores = scores[valid]
        records = [r for r, v in zip(records, valid) if v]
        n = X.shape[0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.tsne_perplexity, max(5, n // 4)),
        random_state=args.tsne_random_state,
        init="pca",
    )
    X_tsne = tsne.fit_transform(X_scaled)

    out_dir = args.out_dir or scores_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_weights_only" if args.weights_only else ""
    title_base = "GST weights only" if args.weights_only else "GST weights + score"

    # Plot 1: t-SNE colored by cluster
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=20,
    )
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.set_title(f"t-SNE of {title_base} (colored by K-means cluster)")
    plt.colorbar(scatter, ax=ax1, label="Cluster")
    plt.tight_layout()
    fig1.savefig(out_dir / f"tsne_by_cluster{suffix}.png", dpi=150)
    plt.close(fig1)

    # Plot 2: t-SNE colored by trustworthiness score
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sc2 = ax2.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=scores,
        cmap="viridis",
        alpha=0.7,
        s=20,
    )
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title(f"t-SNE of {title_base} (colored by trustworthiness score)")
    plt.colorbar(sc2, ax=ax2, label="Trustworthiness score")
    plt.tight_layout()
    fig2.savefig(out_dir / f"tsne_by_score{suffix}.png", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    main()
