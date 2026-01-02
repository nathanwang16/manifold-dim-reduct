"""
Phase 2.3: Static Visualizations

Generates static PNG/HTML visualization files for:
1. 2D scatter plots with labels and centroids
2. PCA variance explained plot
3. Silhouette score bar charts
4. Pairwise distance heatmaps
5. Hierarchical clustering dendrograms
6. Multi-method comparison grid

Usage:
    python static_visualizations.py --embeddings embeddings/ --analysis analysis/
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Add parent directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger, LogTimer, log_metrics, configure_logging

# Initialize logger
logger = get_logger(__name__)

# Color palette for 18 classes
COLORS_18 = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1'
]


def load_embeddings(embeddings_dir: Path) -> dict:
    """Load all embedding files from directory."""
    embeddings = {}
    for f in embeddings_dir.glob("*.npy"):
        if 'labels' not in f.name and 'indices' not in f.name and 'variance' not in f.name:
            embeddings[f.stem] = np.load(f)
    logger.debug(f"Loaded {len(embeddings)} embedding files")
    return embeddings


def load_analysis_results(analysis_dir: Path) -> dict:
    """Load cluster analysis results."""
    results_file = analysis_dir / "cluster_analysis_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    logger.warning(f"Analysis results not found at {results_file}")
    return {}


def plot_scatter_with_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    show_hulls: bool = False,
    subsample: int = None
) -> None:
    """
    Create and save scatter plot with centroids.

    Args:
        embeddings: 2D embeddings (n_samples, 2)
        labels: Class labels
        title: Plot title
        output_path: Path to save figure
        show_hulls: Whether to draw convex hulls
        subsample: Optional subsampling for large datasets
    """
    logger.info(f"Creating scatter plot: {title}")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Subsample if needed
    if subsample and len(embeddings) > subsample:
        np.random.seed(42)
        idx = np.random.choice(len(embeddings), subsample, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        logger.debug(f"Subsampled to {subsample} points")

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        color = COLORS_18[int(label) - 1]

        # Plot points
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=color,
            s=5,
            alpha=0.5,
            label=f'Label {label}'
        )

        # Compute and plot centroid
        centroid = embeddings[mask].mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1],
            c=color,
            s=200,
            marker='X',
            edgecolors='black',
            linewidths=1.5
        )
        ax.annotate(
            str(label),
            (centroid[0], centroid[1]),
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='bottom',
            xytext=(0, 5),
            textcoords='offset points'
        )

        # Draw convex hull
        if show_hulls:
            points = embeddings[mask]
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(
                            points[simplex, 0],
                            points[simplex, 1],
                            c=color,
                            linestyle='--',
                            alpha=0.5,
                            linewidth=0.8
                        )
                except Exception as e:
                    logger.debug(f"Failed to compute hull for label {label}: {e}")

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Create legend with smaller markers
    handles = [mpatches.Patch(color=COLORS_18[i], label=f'L{i+1}') for i in range(18)]
    ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_pca_variance(
    embeddings_dir: Path,
    output_path: Path
) -> None:
    """Plot PCA explained variance."""
    variance_file = embeddings_dir / "pca_variance_ratio.npy"
    if not variance_file.exists():
        logger.warning("PCA variance file not found")
        return

    logger.info("Creating PCA variance plot")
    variance = np.load(variance_file)
    cumulative = np.cumsum(variance)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Individual variance
    ax1.bar(range(1, len(variance) + 1), variance, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance')
    ax1.set_xlim(0.5, min(20.5, len(variance) + 0.5))

    # Cumulative variance
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'o-', color='steelblue')
    ax2.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlim(0.5, min(50.5, len(cumulative) + 0.5))
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_silhouette_comparison(
    analysis_results: dict,
    output_path: Path
) -> None:
    """Plot silhouette scores for all methods."""
    summary = analysis_results.get('comparison_summary', [])
    if not summary:
        logger.warning("No comparison summary found")
        return

    logger.info("Creating silhouette comparison plot")
    methods = [s['method'] for s in summary]
    silhouettes = [s['silhouette'] for s in summary]
    aris = [s['ari'] for s in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette scores
    bars1 = ax1.barh(methods, silhouettes, color='steelblue')
    ax1.set_xlabel('Silhouette Score')
    ax1.set_title('Silhouette Score by Method')
    ax1.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars1, silhouettes):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    # ARI scores
    bars2 = ax2.barh(methods, aris, color='coral')
    ax2.set_xlabel('Adjusted Rand Index')
    ax2.set_title('Adjusted Rand Index by Method')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    for bar, val in zip(bars2, aris):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_distance_heatmap(
    analysis_dir: Path,
    method_name: str,
    output_path: Path
) -> None:
    """Plot pairwise distance heatmap."""
    dist_file = analysis_dir / f"{method_name}_distance_matrix.npy"
    if not dist_file.exists():
        logger.debug(f"Distance matrix not found for {method_name}")
        return

    logger.info(f"Creating distance heatmap for {method_name}")
    distance_matrix = np.load(dist_file)
    labels = [str(i) for i in range(1, 19)]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        distance_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis_r',
        annot=True,
        fmt='.2f',
        annot_kws={'size': 6},
        ax=ax
    )

    ax.set_xlabel('Label')
    ax.set_ylabel('Label')
    ax.set_title(f'Pairwise Centroid Distances: {method_name}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_dendrogram(
    analysis_results: dict,
    method_name: str,
    output_path: Path
) -> None:
    """Plot hierarchical clustering dendrogram."""
    if method_name not in analysis_results:
        logger.debug(f"No analysis results for {method_name}")
        return

    hier_data = analysis_results[method_name].get('hierarchical', {})
    linkage_matrix = hier_data.get('linkage', [])

    if not linkage_matrix:
        logger.debug(f"No linkage data for {method_name}")
        return

    logger.info(f"Creating dendrogram for {method_name}")
    Z = np.array(linkage_matrix)
    labels = [str(i) for i in range(1, 19)]

    fig, ax = plt.subplots(figsize=(12, 6))

    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=0,
        leaf_font_size=10,
        ax=ax
    )

    ax.set_xlabel('Label')
    ax.set_ylabel('Distance')
    ax.set_title(f'Hierarchical Clustering of Labels: {method_name}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_method_comparison_grid(
    embeddings: dict,
    labels: np.ndarray,
    output_path: Path,
    subsample: int = 10000
) -> None:
    """Create grid of all embedding methods for comparison."""
    n_methods = len(embeddings)
    if n_methods == 0:
        logger.warning("No embeddings found for comparison grid")
        return

    logger.info(f"Creating method comparison grid for {n_methods} methods")

    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_methods == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    # Subsample for faster plotting
    if subsample and len(labels) > subsample:
        np.random.seed(42)
        idx = np.random.choice(len(labels), subsample, replace=False)
    else:
        idx = np.arange(len(labels))

    for i, (name, emb) in enumerate(embeddings.items()):
        row, col = i // cols, i % cols
        ax = axes[row][col]

        emb_sub = emb[idx]
        if emb_sub.shape[1] > 2:
            emb_sub = emb_sub[:, :2]
        labels_sub = labels[idx]

        for label in np.unique(labels_sub):
            mask = labels_sub == label
            color = COLORS_18[int(label) - 1]
            ax.scatter(
                emb_sub[mask, 0],
                emb_sub[mask, 1],
                c=color,
                s=2,
                alpha=0.4
            )

        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for i in range(n_methods, rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].axis('off')

    plt.suptitle('Embedding Method Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_all_visualizations(
    embeddings_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    labels: np.ndarray
) -> None:
    """Generate all static visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with LogTimer(logger, "Loading data"):
        embeddings = load_embeddings(embeddings_dir)
        analysis_results = load_analysis_results(analysis_dir)

    figures_generated = 0

    with LogTimer(logger, "Generating all visualizations"):
        # PCA variance plot
        plot_pca_variance(embeddings_dir, output_dir / "pca_variance.png")
        figures_generated += 1

        # Method comparison grid
        plot_method_comparison_grid(embeddings, labels, output_dir / "method_comparison.png")
        figures_generated += 1

        # Silhouette comparison
        plot_silhouette_comparison(analysis_results, output_dir / "silhouette_comparison.png")
        figures_generated += 1

        # Per-method visualizations
        for name, emb in embeddings.items():
            if emb.shape[1] > 2:
                emb = emb[:, :2]

            # Scatter plot
            plot_scatter_with_centroids(
                emb, labels, f'2D Embedding: {name}',
                output_dir / f"{name}_scatter.png",
                subsample=20000
            )
            figures_generated += 1

            # Distance heatmap
            plot_distance_heatmap(analysis_dir, name, output_dir / f"{name}_distances.png")
            figures_generated += 1

            # Dendrogram
            plot_dendrogram(analysis_results, name, output_dir / f"{name}_dendrogram.png")
            figures_generated += 1

    log_metrics(logger, {
        "figures_generated": figures_generated,
        "output_dir": str(output_dir)
    }, message="Visualization generation complete")

    logger.info(f"All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate static visualizations")
    parser.add_argument("--embeddings", type=str, default="phase2_manifold/embeddings",
                        help="Directory containing embeddings")
    parser.add_argument("--analysis", type=str, default="phase2_manifold/analysis",
                        help="Directory containing analysis results")
    parser.add_argument("--labels", type=str, default="phase2_manifold/embeddings/labels.npy",
                        help="Path to labels .npy file")
    parser.add_argument("--output", type=str, default="phase2_manifold/figures",
                        help="Output directory for figures")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)

    embeddings_dir = Path(args.embeddings)
    analysis_dir = Path(args.analysis)
    output_dir = Path(args.output)

    # Load labels
    with LogTimer(logger, f"Loading labels from {args.labels}"):
        labels = np.load(args.labels)
        logger.info(f"Loaded {len(labels)} labels")

    # Generate all visualizations
    generate_all_visualizations(embeddings_dir, analysis_dir, output_dir, labels)

    logger.info("Static visualization generation complete!")


if __name__ == "__main__":
    main()
