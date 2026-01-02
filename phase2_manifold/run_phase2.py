"""
Phase 2 Pipeline Runner

Orchestrates the complete Phase 2 manifold learning pipeline:
1. Feature extraction (k-mer, positional, dinucleotide)
2. Dimensionality reduction (PCA, UMAP, PHATE)
3. Cluster analysis (silhouette, ARI, hierarchical)
4. Visualization generation

Usage:
    # Use full visualization data (default - 171,699 sequences):
    python run_phase2.py

    # Use demo data for fast testing:
    python run_phase2.py --use-demo-data

    # Specify custom data files:
    python run_phase2.py --sequences trainsequences.csv --labels trainlabels.csv

    # With subsampling for faster testing:
    python run_phase2.py --subsample 50000

    # Skip steps (use existing features/embeddings):
    python run_phase2.py --skip-extraction --skip-reduction

    # Launch interactive dashboard:
    python run_phase2.py --dashboard
"""

import sys
import argparse
import json
import subprocess
from pathlib import Path

# Add parent directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger, LogTimer, log_metrics, configure_logging, setup_exception_logging

# Initialize logger
logger = get_logger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_step(script: str, args: list, step_name: str) -> bool:
    """Run a pipeline step and handle errors."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*60}")

    cmd = [sys.executable, script] + args
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        with LogTimer(logger, step_name):
            result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"Completed: {step_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {step_name}")
        logger.error(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in {step_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 manifold learning pipeline")
    parser.add_argument("--sequences", type=str, default=None, help="Path to sequences CSV (overrides config)")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV (overrides config)")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--use-demo-data", action="store_true",
                        help="Use demo data from config (small dataset for testing)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N points for faster processing")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction (use existing features)")
    parser.add_argument("--skip-reduction", action="store_true",
                        help="Skip dimensionality reduction (use existing embeddings)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch interactive dashboard after processing")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)
    setup_exception_logging(logger)

    logger.info("="*60)
    logger.info("PHASE 2 MANIFOLD LEARNING PIPELINE")
    logger.info("="*60)

    # Load config and determine data source
    config = load_config(args.config)
    base_dir = Path(__file__).parent

    # Determine which data to use (viz data is default)
    if args.sequences and args.labels:
        # User explicitly provided sequences/labels
        sequences_file = args.sequences
        labels_file = args.labels
        use_viz_data = False
        logger.info(f"Using command-line provided data: {sequences_file}")
    elif args.use_demo_data:
        # Use demo data
        sequences_file = config['data']['train_sequences']
        labels_file = config['data']['train_labels']
        use_viz_data = False
        logger.info(f"Using demo data from config: {sequences_file}")
    else:
        # Default: use viz data (full dataset)
        sequences_file = config['data']['viz_sequences']
        labels_file = config['data']['viz_labels']
        use_viz_data = True
        logger.info(f"Using full visualization data from config: {sequences_file}")

    log_metrics(logger, {
        "sequences_file": sequences_file,
        "labels_file": labels_file,
        "use_viz_data": use_viz_data,
        "subsample": args.subsample or "none",
        "skip_extraction": args.skip_extraction,
        "skip_reduction": args.skip_reduction
    }, message="Pipeline configuration")

    steps_completed = 0
    steps_failed = 0

    with LogTimer(logger, "Full Phase 2 Pipeline"):
        # Step 1: Feature Extraction
        if not args.skip_extraction:
            extraction_args = [
                "--input", sequences_file,
                "--labels", labels_file,
                "--config", args.config,
                "--output", config['output']['features_dir'],
                "--log-dir", args.log_dir,
                "--n-jobs", str(config.get('n_jobs', -1)),
                "--batch-size", str(config.get('feature_extraction_batch_size', 1000))
            ]
            if run_step(str(base_dir / "feature_extraction.py"), extraction_args,
                       "Feature Extraction"):
                steps_completed += 1
            else:
                steps_failed += 1
                logger.error("Pipeline failed at feature extraction")
                return 1
        else:
            logger.info("Skipping feature extraction (--skip-extraction)")

        # Step 2: Dimensionality Reduction
        if not args.skip_reduction:
            features_path = Path(config['output']['features_dir']) / f"kmer_{config['phase2']['kmer_k']}_features.npy"
            labels_path = Path(config['output']['features_dir']) / "labels.npy"

            reduction_args = [
                "--features", str(features_path),
                "--labels", str(labels_path),
                "--config", args.config,
                "--output", config['output']['embeddings_dir'],
                "--log-dir", args.log_dir
            ]
            if args.subsample:
                reduction_args.extend(["--subsample", str(args.subsample)])

            if run_step(str(base_dir / "dimensionality_reduction.py"), reduction_args,
                       "Dimensionality Reduction"):
                steps_completed += 1
            else:
                steps_failed += 1
                logger.error("Pipeline failed at dimensionality reduction")
                return 1
        else:
            logger.info("Skipping dimensionality reduction (--skip-reduction)")

        # Step 3: Cluster Analysis
        labels_path = Path(config['output']['embeddings_dir']) / "labels.npy"
        analysis_args = [
            "--embeddings", config['output']['embeddings_dir'],
            "--labels", str(labels_path),
            "--config", args.config,
            "--output", config['output']['analysis_dir'],
            "--log-dir", args.log_dir
        ]
        if run_step(str(base_dir / "cluster_analysis.py"), analysis_args,
                   "Cluster Analysis"):
            steps_completed += 1
        else:
            steps_failed += 1
            logger.error("Pipeline failed at cluster analysis")
            return 1

        # Step 4: Static Visualizations
        viz_args = [
            "--embeddings", config['output']['embeddings_dir'],
            "--analysis", config['output']['analysis_dir'],
            "--labels", str(labels_path),
            "--output", config['output']['figures_dir'],
            "--log-dir", args.log_dir
        ]
        if run_step(str(base_dir / "static_visualizations.py"), viz_args,
                   "Static Visualizations"):
            steps_completed += 1
        else:
            steps_failed += 1
            logger.error("Pipeline failed at visualization generation")
            return 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2 PIPELINE COMPLETE")
    logger.info(f"{'='*60}")

    log_metrics(logger, {
        "steps_completed": steps_completed,
        "steps_failed": steps_failed,
        "features_dir": config['output']['features_dir'],
        "embeddings_dir": config['output']['embeddings_dir'],
        "analysis_dir": config['output']['analysis_dir'],
        "figures_dir": config['output']['figures_dir']
    }, message="Pipeline summary")

    logger.info(f"Features: {config['output']['features_dir']}")
    logger.info(f"Embeddings: {config['output']['embeddings_dir']}")
    logger.info(f"Analysis: {config['output']['analysis_dir']}")
    logger.info(f"Figures: {config['output']['figures_dir']}")

    # Optional: Launch dashboard
    if args.dashboard:
        logger.info("\nLaunching interactive dashboard...")
        dashboard_args = [
            "--embeddings", config['output']['embeddings_dir'],
            "--analysis", config['output']['analysis_dir'],
            "--labels", str(labels_path),
            "--log-dir", args.log_dir
        ]
        run_step(str(base_dir / "visualization_dashboard.py"), dashboard_args,
                "Interactive Dashboard")

    return 0


if __name__ == "__main__":
    sys.exit(main())
