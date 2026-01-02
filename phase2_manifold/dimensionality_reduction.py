"""
Phase 2.2: Dimensionality Reduction Methods

Compares three dimensionality reduction approaches:
1. UMAP - Uniform Manifold Approximation and Projection
2. PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
3. PCA - Principal Component Analysis (baseline)

Usage:
    python dimensionality_reduction.py --features features/kmer_5_features.npy
"""

import sys
import json
import argparse
import os
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import phate

# Add parent directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger, LogTimer, log_metrics, configure_logging

# Initialize logger
logger = get_logger(__name__)


def setup_parallel_environment(n_jobs: int = -1) -> int:
    """
    Configure environment variables for optimal parallel performance on Mac M1.

    Args:
        n_jobs: Requested number of parallel jobs (-1 = all cores)

    Returns:
        Actual number of cores to use
    """
    # Get available CPU cores
    available_cores = multiprocessing.cpu_count()
    logger.info(f"Available CPU cores: {available_cores}")

    # Determine actual number of jobs
    if n_jobs == -1:
        actual_jobs = available_cores
    elif n_jobs <= 0 or n_jobs > available_cores:
        logger.warning(f"n_jobs={n_jobs} invalid, using {available_cores}")
        actual_jobs = available_cores
    else:
        actual_jobs = n_jobs

    logger.info(f"Using {actual_jobs} parallel jobs")

    # Set environment variables for optimal BLAS performance
    os.environ['OMP_NUM_THREADS'] = str(actual_jobs)
    os.environ['OPENBLAS_NUM_THREADS'] = str(actual_jobs)
    os.environ['MKL_NUM_THREADS'] = str(actual_jobs)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(actual_jobs)

    logger.debug(f"Set OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    logger.debug(f"Set OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']}")
    logger.debug(f"Set MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']}")
    logger.debug(f"Set VECLIB_MAXIMUM_THREADS={os.environ['VECLIB_MAXIMUM_THREADS']}")

    # Check if we're running on Apple Silicon
    try:
        import platform
        if platform.machine() in ('arm64', 'aarch64'):
            logger.info("Detected Apple Silicon architecture")
            # Accelerate framework is automatically used by numpy/blas
    except Exception as e:
        logger.debug(f"Could not detect architecture: {e}")

    return actual_jobs


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_pca(
    features: np.ndarray,
    n_components: int = 50,
    return_full: bool = True
) -> dict:
    """
    Run PCA dimensionality reduction.

    Args:
        features: Input feature matrix (n_samples, n_features)
        n_components: Number of components to compute
        return_full: If True, return explained variance info

    Returns:
        Dictionary with embeddings and optional variance info
    """
    logger.info(f"Running PCA with {n_components} components...")
    logger.debug(f"Input shape: {features.shape}")

    with LogTimer(logger, "PCA fitting"):
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(features_scaled)

    result = {
        'embeddings': embeddings,
        'embeddings_2d': embeddings[:, :2],
        'embeddings_3d': embeddings[:, :3] if n_components >= 3 else None,
    }

    if return_full:
        result['explained_variance_ratio'] = pca.explained_variance_ratio_
        result['cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)
        result['components'] = pca.components_

        log_metrics(logger, {
            "variance_2pc": float(result['cumulative_variance'][1]),
            "variance_10pc": float(result['cumulative_variance'][min(9, n_components-1)]),
            "n_components": n_components
        }, message="PCA results")

        logger.info(f"Variance explained by first 2 PCs: {result['cumulative_variance'][1]:.3f}")
        logger.info(f"Variance explained by first 10 PCs: {result['cumulative_variance'][min(9, n_components-1)]:.3f}")

    return result


def run_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    n_jobs: int = -1,
    metric: str = 'euclidean',
    random_state: Optional[int] = None 
) -> dict:
    """
    Run UMAP dimensionality reduction.
    Using Euclidean distance metric.
 
    Args:
        features: Input feature matrix
        n_neighbors: Size of local neighborhood
        min_dist: Minimum distance between points
        n_components: Target dimensionality
        n_jobs: Number of parallel jobs (-1 = all cores)
        metric: Distance metric
        random_state: Random seed

    Returns:
        Dictionary with embeddings
    """
    #TODO may need to revise the eucidian metrics option
    logger.info(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_jobs={n_jobs})")
    logger.debug(f"Input shape: {features.shape}")

    with LogTimer(logger, f"UMAP n={n_neighbors} d={min_dist}"):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=True
        )
        embeddings = reducer.fit_transform(features)

    log_metrics(logger, {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": n_components,
        "n_jobs": n_jobs,
        "output_shape": list(embeddings.shape)
    }, message="UMAP results")

    return {
        'embeddings': embeddings,
        'embeddings_2d': embeddings if n_components == 2 else embeddings[:, :2],
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'n_jobs': n_jobs
    }


def run_phate(
    features: np.ndarray,
    knn: int = 10,
    decay: int = 20,
    n_components: int = 2,
    n_jobs: int = -1,
    random_state: int = 42
) -> dict:
    """
    Run PHATE dimensionality reduction.

    Args:
        features: Input feature matrix
        knn: Number of nearest neighbors
        decay: Decay parameter for kernel
        n_components: Target dimensionality
        n_jobs: Number of parallel jobs (-1 = all cores)
        random_state: Random seed

    Returns:
        Dictionary with embeddings
    """
    logger.info(f"Running PHATE (knn={knn}, decay={decay}, n_jobs={n_jobs})")
    logger.debug(f"Input shape: {features.shape}")

    with LogTimer(logger, f"PHATE k={knn} d={decay}"):
        phate_op = phate.PHATE(
            knn=knn,
            decay=decay,
            n_components=n_components,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
        embeddings = phate_op.fit_transform(features)

    log_metrics(logger, {
        "knn": knn,
        "decay": decay,
        "n_components": n_components,
        "n_jobs": n_jobs,
        "output_shape": list(embeddings.shape)
    }, message="PHATE results")

    return {
        'embeddings': embeddings,
        'embeddings_2d': embeddings if n_components == 2 else embeddings[:, :2],
        'knn': knn,
        'decay': decay,
        'n_jobs': n_jobs
    }


def run_phate_with_pca(
    features: np.ndarray,
    knn: int = 10,
    decay: int = 20,
    n_components: int = 2,
    n_pcs: int = 50,
    n_jobs: int = -1,
    random_state: int = 42
) -> dict:
    """
    Run PHATE dimensionality reduction with PCA pre-processing.

    Reduces dimensionality with PCA before running PHATE to significantly
    speed up KNN search in high-dimensional feature spaces.

    Args:
        features: Input feature matrix (n_samples, n_features)
        knn: Number of nearest neighbors for PHATE
        decay: Decay parameter for PHATE kernel
        n_components: Target dimensionality for PHATE output
        n_pcs: Number of PCA components to keep before PHATE
        n_jobs: Number of parallel jobs (-1 = all cores)
        random_state: Random seed

    Returns:
        Dictionary with embeddings and metadata
    """
    logger.info(f"Running PHATE with PCA pre-processing (n_pcs={n_pcs}, knn={knn}, decay={decay}, n_jobs={n_jobs})")
    logger.debug(f"Input shape: {features.shape}")

    # Step 1: PCA pre-processing
    with LogTimer(logger, "PCA pre-processing"):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=n_pcs, random_state=random_state)
        features_reduced = pca.fit_transform(features_scaled)

        logger.info(f"Reduced dimensions: {features.shape} -> {features_reduced.shape}")

        # Log variance explained
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        log_metrics(logger, {
            "n_pcs": n_pcs,
            "variance_explained_last_pc": float(cumulative_variance[-1]),
            "variance_explained_first_10_pc": float(cumulative_variance[min(9, n_pcs-1)]),
        }, message="PCA pre-processing metrics")

    # Step 2: PHATE on reduced features
    with LogTimer(logger, f"PHATE on reduced features k={knn} d={decay}"):
        phate_op = phate.PHATE(
            knn=knn,
            decay=decay,
            n_components=n_components,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
        embeddings = phate_op.fit_transform(features_reduced)

    log_metrics(logger, {
        "knn": knn,
        "decay": decay,
        "n_components": n_components,
        "n_pcs": n_pcs,
        "n_jobs": n_jobs,
        "output_shape": list(embeddings.shape)
    }, message="PHATE with PCA results")

    return {
        'embeddings': embeddings,
        'embeddings_2d': embeddings if n_components == 2 else embeddings[:, :2],
        'knn': knn,
        'decay': decay,
        'n_pcs': n_pcs,
        'n_jobs': n_jobs,
        'pca_variance_ratio': pca.explained_variance_ratio_,
        'pca_cumulative_variance': cumulative_variance
    }


def run_all_reductions(
    features: np.ndarray,
    config: dict,
    output_dir: Path,
    labels: Optional[np.ndarray] = None
) -> dict:
    """
    Run all dimensionality reduction methods with configured parameters.

    Args:
        features: Input feature matrix
        config: Configuration dictionary
        output_dir: Directory to save results
        labels: Optional labels for later analysis

    Returns:
        Dictionary with all embeddings
    """
    phase2_config = config['phase2']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parallel processing settings
    n_jobs = phase2_config.get('n_jobs', -1)
    use_pca_preprocessing = phase2_config.get('use_pca_preprocessing', False)
    n_pcs_for_phate = phase2_config.get('n_pcs_for_phate', 50)

    results = {}
    method_count = 0

    # PCA
    with LogTimer(logger, "PCA reduction"):
        pca_result = run_pca(features, n_components=phase2_config['pca_n_components'])
        results['pca'] = pca_result
        np.save(output_dir / "pca_embeddings.npy", pca_result['embeddings'])
        np.save(output_dir / "pca_2d.npy", pca_result['embeddings_2d'])
        np.save(output_dir / "pca_variance_ratio.npy", pca_result['explained_variance_ratio'])
        method_count += 1

    # UMAP with multiple parameter combinations
    umap_params_list = phase2_config['umap_params']
    logger.info(f"Running {len(umap_params_list)} UMAP configurations")

    for i, params in enumerate(umap_params_list):
        with LogTimer(logger, f"UMAP config {i+1}/{len(umap_params_list)}"):
            umap_result = run_umap(
                features,
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                n_jobs=n_jobs,
                random_state=None  # Set to None to enable parallel processing
            )
            key = f"umap_n{params['n_neighbors']}_d{params['min_dist']}"
            results[key] = umap_result
            np.save(output_dir / f"{key}.npy", umap_result['embeddings_2d'])
            method_count += 1

    # PHATE with multiple parameter combinations
    phate_params_list = phase2_config['phate_params']
    logger.info(f"Running {len(phate_params_list)} PHATE configurations")

    for i, params in enumerate(phate_params_list):
        with LogTimer(logger, f"PHATE config {i+1}/{len(phate_params_list)}"):
            if use_pca_preprocessing:
                logger.info(f"Using PCA pre-processing for PHATE (n_pcs={n_pcs_for_phate})")
                phate_result = run_phate_with_pca(
                    features,
                    knn=params['knn'],
                    decay=params['decay'],
                    n_pcs=n_pcs_for_phate,
                    n_jobs=n_jobs
                )
            else:
                phate_result = run_phate(
                    features,
                    knn=params['knn'],
                    decay=params['decay'],
                    n_jobs=n_jobs
                )
            key = f"phate_k{params['knn']}_d{params['decay']}"
            if use_pca_preprocessing:
                key = f"{key}_pca{n_pcs_for_phate}"
            results[key] = phate_result
            np.save(output_dir / f"{key}.npy", phate_result['embeddings_2d'])
            method_count += 1

    # Save labels if provided
    if labels is not None:
        np.save(output_dir / "labels.npy", labels)
        logger.info(f"Saved labels to {output_dir / 'labels.npy'}")

    # Save metadata
    metadata = {
        'feature_shape': list(features.shape),
        'methods': list(results.keys()),
        'config': phase2_config
    }
    with open(output_dir / "reduction_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    log_metrics(logger, {
        "total_methods": method_count,
        "umap_configs": len(umap_params_list),
        "phate_configs": len(phate_params_list),
        "output_dir": str(output_dir)
    }, message="Dimensionality reduction summary")

    logger.info(f"Saved {len(results)} embeddings to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run dimensionality reduction methods")
    parser.add_argument("--features", type=str, required=True, help="Path to features .npy file")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels .npy file")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--output", type=str, default="phase2_manifold/embeddings", help="Output directory")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample N points for faster testing")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)

    config = load_config(args.config)
    output_dir = Path(args.output)

    # Setup parallel processing environment
    n_jobs = config['phase2'].get('n_jobs', -1)
    actual_n_jobs = setup_parallel_environment(n_jobs)

    # Load features
    with LogTimer(logger, f"Loading features from {args.features}"):
        features = np.load(args.features)
        logger.info(f"Features shape: {features.shape}")

    # Log parallelization settings
    log_metrics(logger, {
        "n_jobs_requested": n_jobs,
        "n_jobs_actual": actual_n_jobs,
        "use_pca_preprocessing": config['phase2'].get('use_pca_preprocessing', False),
        "n_pcs_for_phate": config['phase2'].get('n_pcs_for_phate', 50)
    }, message="Parallelization configuration")

    # Load labels if provided
    labels = None
    if args.labels:
        with LogTimer(logger, f"Loading labels from {args.labels}"):
            labels = np.load(args.labels)
            logger.info(f"Labels shape: {labels.shape}")

    # Subsample if requested (for faster testing)
    if args.subsample and args.subsample < len(features):
        logger.info(f"Subsampling {args.subsample} points from {len(features)}")
        np.random.seed(42)
        indices = np.random.choice(len(features), args.subsample, replace=False)
        features = features[indices]
        if labels is not None:
            labels = labels[indices]
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "subsample_indices.npy", indices)
        log_metrics(logger, {"subsample_size": args.subsample}, message="Subsampling")

    # Run all reductions
    with LogTimer(logger, "All dimensionality reductions"):
        results = run_all_reductions(features, config, output_dir, labels)

    logger.info("Dimensionality reduction complete!")


if __name__ == "__main__":
    main()
