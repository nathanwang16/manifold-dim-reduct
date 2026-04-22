"""Phase 4: Sparse Autoencoders on ChromatinCNNAttentionV2 bottleneck.

Pipeline:
    extract_activations.py  — cache bottleneck activations from a trained
                              checkpoint over a balanced val subset.
    sae.py                  — BaseSAE / TopKSAE / L1SAE definitions.
    train_sae.py            — train an SAE on cached activations.
    feature_analysis.py     — per-feature diagnostics: top-k samples, class
                              selectivity, L0, dead-feature rate.
    run_phase4.py           — end-to-end orchestrator.

Outputs a "feature dictionary" (`results/phase4_sae/features.json`) that is
consumed by phase 5 (patching) and phase 8 (motif discovery).
"""

from .sae import BaseSAE, TopKSAE, L1SAE, build_sae

__all__ = ["BaseSAE", "TopKSAE", "L1SAE", "build_sae"]
