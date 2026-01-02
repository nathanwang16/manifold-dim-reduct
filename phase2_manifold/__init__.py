"""
Phase 2: Manifold Learning & Visualization

This module provides tools for manifold-based analysis of DNA sequences:
- Feature extraction (k-mer frequencies, positional profiles, dinucleotides)
- Dimensionality reduction (PCA, UMAP, PHATE)
- Cluster analysis (silhouette, ARI, hierarchical clustering)
- Visualization (static plots and interactive dashboard)
"""

from .feature_extraction import (
    compute_kmer_frequencies,
    compute_positional_kmer_profiles,
    compute_dinucleotide_frequencies,
    generate_kmer_vocabulary
)

from .dimensionality_reduction import (
    run_pca,
    run_umap,
    run_phate
)

from .cluster_analysis import (
    compute_silhouette_scores,
    compute_kmeans_clustering,
    compute_label_centroids,
    compute_pairwise_distances,
    predict_confusions
)

__all__ = [
    'compute_kmer_frequencies',
    'compute_positional_kmer_profiles',
    'compute_dinucleotide_frequencies',
    'generate_kmer_vocabulary',
    'run_pca',
    'run_umap',
    'run_phate',
    'compute_silhouette_scores',
    'compute_kmeans_clustering',
    'compute_label_centroids',
    'compute_pairwise_distances',
    'predict_confusions'
]
