"""
Phase 2.3: Quantitative Cluster Analysis

Performs:
1. Silhouette score computation for 18-class labeling
2. Adjusted Rand Index between k-means and true labels
3. Intra-class cohesion analysis
4. Pairwise label centroid distances
5. Hierarchical clustering of label relationships
6. Confusion prediction based on manifold proximity

Usage:
    python cluster_analysis.py --embeddings embeddings/ --labels labels.npy
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Add parent directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger, LogTimer, log_metrics, configure_logging

# Initialize logger
logger = get_logger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def compute_silhouette_scores(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Compute silhouette scores for the labeled embedding.

    Args:
        embeddings: 2D embeddings (n_samples, 2)
        labels: True class labels

    Returns:
        Dictionary with overall and per-class silhouette scores
    """
    logger.info("Computing silhouette scores...")
    logger.debug(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")

    with LogTimer(logger, "Silhouette computation"):
        # Overall silhouette score
        overall_score = silhouette_score(embeddings, labels)

        # Per-sample silhouette values
        sample_scores = silhouette_samples(embeddings, labels)

        # Per-class average silhouette scores
        unique_labels = np.unique(labels)
        per_class_scores = {}
        for label in unique_labels:
            mask = labels == label
            per_class_scores[int(label)] = float(np.mean(sample_scores[mask]))

    # Identify well-clustered vs dispersed classes
    sorted_classes = sorted(per_class_scores.items(), key=lambda x: x[1], reverse=True)
    well_clustered = [c[0] for c in sorted_classes[:5]]
    dispersed = [c[0] for c in sorted_classes[-5:]]

    log_metrics(logger, {
        "overall_silhouette": float(overall_score),
        "best_class": sorted_classes[0][0],
        "best_score": float(sorted_classes[0][1]),
        "worst_class": sorted_classes[-1][0],
        "worst_score": float(sorted_classes[-1][1])
    }, message="Silhouette scores")

    logger.info(f"Overall silhouette score: {overall_score:.4f}")
    logger.info(f"Best clustered classes: {well_clustered}")
    logger.info(f"Most dispersed classes: {dispersed}")

    return {
        'overall': float(overall_score),
        'per_class': per_class_scores,
        'well_clustered': well_clustered,
        'dispersed': dispersed,
        'sample_scores': sample_scores
    }


def compute_kmeans_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = 18,
    random_state: int = 42
) -> dict:
    """
    Run k-means and compare to true labels via Adjusted Rand Index.

    Args:
        embeddings: 2D embeddings
        labels: True class labels
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Dictionary with clustering results and ARI
    """
    logger.info(f"Running k-means with {n_clusters} clusters...")

    with LogTimer(logger, "K-means clustering"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        predicted_clusters = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(labels, predicted_clusters)

    log_metrics(logger, {
        "n_clusters": n_clusters,
        "adjusted_rand_index": float(ari),
        "inertia": float(kmeans.inertia_)
    }, message="K-means results")

    logger.info(f"Adjusted Rand Index: {ari:.4f}")

    return {
        'predicted_clusters': predicted_clusters,
        'cluster_centers': kmeans.cluster_centers_,
        'adjusted_rand_index': float(ari),
        'inertia': float(kmeans.inertia_)
    }


def compute_label_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Compute centroid position for each label class.

    Args:
        embeddings: 2D embeddings
        labels: True class labels

    Returns:
        Dictionary with centroids and per-class statistics
    """
    logger.info("Computing label centroids...")

    unique_labels = np.unique(labels)
    centroids = {}
    class_stats = {}

    for label in unique_labels:
        mask = labels == label
        points = embeddings[mask]

        centroid = points.mean(axis=0)
        centroids[int(label)] = centroid.tolist()

        # Compute intra-class statistics
        distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
        class_stats[int(label)] = {
            'count': int(mask.sum()),
            'mean_distance': float(distances_to_centroid.mean()),
            'std_distance': float(distances_to_centroid.std()),
            'max_distance': float(distances_to_centroid.max())
        }

    logger.debug(f"Computed centroids for {len(centroids)} classes")
    return {
        'centroids': centroids,
        'class_stats': class_stats
    }


def compute_pairwise_distances(centroids: dict) -> np.ndarray:
    """
    Compute distance matrix between all label centroids.

    Args:
        centroids: Dictionary mapping label to centroid coordinates

    Returns:
        18x18 distance matrix
    """
    logger.info("Computing pairwise centroid distances...")

    labels = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in labels])

    distance_matrix = squareform(pdist(centroid_matrix, metric='euclidean'))

    logger.debug(f"Distance matrix shape: {distance_matrix.shape}")
    return distance_matrix


def compute_hierarchical_clustering(distance_matrix: np.ndarray) -> dict:
    """
    Perform hierarchical clustering on label centroids.

    Args:
        distance_matrix: 18x18 distance matrix

    Returns:
        Dictionary with linkage info and cluster assignments
    """
    logger.info("Computing hierarchical clustering of labels...")

    with LogTimer(logger, "Hierarchical clustering"):
        # Compute linkage
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, method='ward')

        # Cut tree at different levels to get super-groups
        clusters_3 = fcluster(Z, t=3, criterion='maxclust')
        clusters_6 = fcluster(Z, t=6, criterion='maxclust')
        clusters_9 = fcluster(Z, t=9, criterion='maxclust')

    logger.info(f"Created cluster assignments at 3, 6, and 9 levels")

    return {
        'linkage': Z.tolist(),
        'clusters_3': clusters_3.tolist(),
        'clusters_6': clusters_6.tolist(),
        'clusters_9': clusters_9.tolist()
    }


def predict_confusions(distance_matrix: np.ndarray, n_pairs: int = 10) -> list:
    """
    Predict which label pairs are most likely to be confused based on proximity.

    Args:
        distance_matrix: 18x18 distance matrix
        n_pairs: Number of closest pairs to return

    Returns:
        List of (label_i, label_j, distance) tuples
    """
    logger.info("Predicting potential confusion pairs...")

    n_labels = distance_matrix.shape[0]
    pairs = []

    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            # Labels are 1-indexed
            pairs.append((i + 1, j + 1, distance_matrix[i, j]))

    # Sort by distance (closest first = most likely confused)
    pairs.sort(key=lambda x: x[2])

    logger.info(f"Top {n_pairs} most likely confused pairs:")
    for i, (l1, l2, dist) in enumerate(pairs[:n_pairs]):
        logger.info(f"  {i+1}. Labels {l1} & {l2}: distance = {dist:.4f}")

    return pairs[:n_pairs]


def analyze_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    name: str,
    output_dir: Path
) -> dict:
    """
    Run full analysis pipeline on a single embedding.

    Args:
        embeddings: 2D embeddings
        labels: True class labels
        name: Name of the embedding method
        output_dir: Output directory

    Returns:
        Dictionary with all analysis results
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Analyzing embedding: {name}")
    logger.info(f"{'='*50}")

    results = {'name': name}

    with LogTimer(logger, f"Full analysis of {name}"):
        # Silhouette analysis
        silhouette_results = compute_silhouette_scores(embeddings, labels)
        results['silhouette'] = {
            'overall': silhouette_results['overall'],
            'per_class': silhouette_results['per_class'],
            'well_clustered': silhouette_results['well_clustered'],
            'dispersed': silhouette_results['dispersed']
        }

        # K-means clustering
        kmeans_results = compute_kmeans_clustering(embeddings, labels)
        results['kmeans'] = {
            'adjusted_rand_index': kmeans_results['adjusted_rand_index'],
            'inertia': kmeans_results['inertia']
        }

        # Label centroids
        centroid_results = compute_label_centroids(embeddings, labels)
        results['centroids'] = centroid_results['centroids']
        results['class_stats'] = centroid_results['class_stats']

        # Pairwise distances
        distance_matrix = compute_pairwise_distances(centroid_results['centroids'])
        np.save(output_dir / f"{name}_distance_matrix.npy", distance_matrix)
        results['distance_matrix_file'] = f"{name}_distance_matrix.npy"

        # Hierarchical clustering
        hier_results = compute_hierarchical_clustering(distance_matrix)
        results['hierarchical'] = hier_results

        # Confusion prediction
        confusion_pairs = predict_confusions(distance_matrix)
        results['predicted_confusions'] = [
            {'label_1': p[0], 'label_2': p[1], 'distance': float(p[2])}
            for p in confusion_pairs
        ]

    log_metrics(logger, {
        "method": name,
        "silhouette": results['silhouette']['overall'],
        "ari": results['kmeans']['adjusted_rand_index']
    }, message=f"Analysis complete: {name}")

    return results


def run_full_analysis(
    embeddings_dir: Path,
    labels: np.ndarray,
    output_dir: Path
) -> dict:
    """
    Run analysis on all embeddings in directory.

    Args:
        embeddings_dir: Directory containing embedding .npy files
        labels: True class labels
        output_dir: Output directory for results

    Returns:
        Dictionary with all analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all embedding files
    embedding_files = list(embeddings_dir.glob("*.npy"))
    embedding_files = [f for f in embedding_files if 'labels' not in f.name
                       and 'indices' not in f.name
                       and 'variance' not in f.name
                       and 'distance' not in f.name]

    logger.info(f"Found {len(embedding_files)} embedding files")

    all_results = {}
    comparison_summary = []

    with LogTimer(logger, "Full cluster analysis"):
        for emb_file in embedding_files:
            name = emb_file.stem
            embeddings = np.load(emb_file)

            # Ensure 2D
            if embeddings.ndim > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            if embeddings.shape[1] > 2:
                embeddings = embeddings[:, :2]

            results = analyze_embedding(embeddings, labels, name, output_dir)
            all_results[name] = results

            comparison_summary.append({
                'method': name,
                'silhouette': results['silhouette']['overall'],
                'ari': results['kmeans']['adjusted_rand_index']
            })

    # Create comparison summary
    logger.info(f"\n{'='*50}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*50}")

    comparison_summary.sort(key=lambda x: x['silhouette'], reverse=True)
    logger.info("\nRanked by Silhouette Score:")
    for i, item in enumerate(comparison_summary):
        logger.info(f"  {i+1}. {item['method']}: silhouette={item['silhouette']:.4f}, ARI={item['ari']:.4f}")

    all_results['comparison_summary'] = comparison_summary

    # Log best method
    if comparison_summary:
        best = comparison_summary[0]
        log_metrics(logger, {
            "best_method": best['method'],
            "best_silhouette": best['silhouette'],
            "best_ari": best['ari'],
            "total_methods": len(comparison_summary)
        }, message="Best performing method")

    # Save all results
    with open(output_dir / "cluster_analysis_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    logger.info(f"\nResults saved to {output_dir / 'cluster_analysis_results.json'}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run cluster analysis on embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help="Directory containing embeddings")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels .npy file")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--output", type=str, default="phase2_manifold/analysis", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)

    embeddings_dir = Path(args.embeddings)
    output_dir = Path(args.output)

    # Load labels
    with LogTimer(logger, f"Loading labels from {args.labels}"):
        labels = np.load(args.labels)
        logger.info(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels)}")

    # Run analysis
    results = run_full_analysis(embeddings_dir, labels, output_dir)

    logger.info("Cluster analysis complete!")


if __name__ == "__main__":
    main()
