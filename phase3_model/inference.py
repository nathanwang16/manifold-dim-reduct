#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for ChromatinCNN models trained with phase3_model/train.py
Outputs predictions.csv with just the predicted labels (1-18)
"""

# ==============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ==============================================================================
CONFIG = {
    # Model checkpoint path
    'checkpoint_path': '/content/drive/MyDrive/chromatin_model_checkpoints_improved/best_model_improved.pt',
    
    # Data paths
    'test_sequences_csv': './data/testsequences.csv',
    'output_csv': 'predictions.csv',
    
    # Inference settings
    'batch_size': 1024,
    'use_rc_tta': True,  # Reverse complement test-time augmentation (improves accuracy)
    
    # Device (auto-detected if None)
    'device': None,  # Options: 'cuda', 'mps', 'cpu', or None for auto-detect
}

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import shared model definitions
# Make sure pythonpath allows importing from phase3_model.model_defs
try:
    from phase3_model.model_defs import ChromatinCNNAttention, ResidualBlock
except ImportError:
    # Fallback if running as script from within the folder
    from model_defs import ChromatinCNNAttention, ResidualBlock

# ==============================================================================
# SETUP
# ==============================================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = CONFIG['device'] if CONFIG['device'] else detect_device()
print(f"Using device: {device}")

# ==============================================================================
# DATA UTILITIES
# ==============================================================================

def one_hot_encode(sequence: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoded array."""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((200, 4), dtype=np.float32)
    for i, base in enumerate(sequence):
        if base in base_to_idx:
            one_hot[i, base_to_idx[base]] = 1.0
        else:
            one_hot[i, :] = 0.25
    return one_hot


def compute_engineered_features(sequence: str) -> np.ndarray:
    """Compute 5 engineered features: gc_content, cpg_ratio, entropy, max_run_len, repeat_density"""
    s = sequence.upper()
    n = len(s) if len(s) > 0 else 1

    a = s.count("A")
    c = s.count("C")
    g = s.count("G")
    t = s.count("T")
    total = a + c + g + t
    if total == 0:
        return np.array([0.5, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)

    gc_content = (g + c) / total

    cpg_obs = sum(1 for i in range(len(s) - 1) if s[i] == "C" and s[i + 1] == "G")
    c_freq = c / total
    g_freq = g / total
    cpg_exp = max(1e-8, (len(s) - 1) * c_freq * g_freq)
    cpg_ratio = float(cpg_obs) / float(cpg_exp)

    p = np.array([a, c, g, t], dtype=np.float32) / float(total)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    max_run = 1
    repeat_positions = 0
    run_len = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] and s[i] in "ACGT":
            run_len += 1
        else:
            max_run = max(max_run, run_len)
            if run_len >= 3:
                repeat_positions += run_len
            run_len = 1
    max_run = max(max_run, run_len)
    if run_len >= 3:
        repeat_positions += run_len

    repeat_density = repeat_positions / max(1, len(s))

    return np.array([gc_content, cpg_ratio, entropy, float(max_run), repeat_density], dtype=np.float32)


class TestDataset(Dataset):
    def __init__(self, sequences_file, use_engineered_features=False):
        self.sequences_file = Path(sequences_file)
        self.use_engineered_features = use_engineered_features

        print(f"Loading test sequences from {self.sequences_file}")
        sequences_df = pd.read_csv(self.sequences_file, header=None)
        self.sequences = sequences_df.iloc[:, 0].values if sequences_df.shape[1] == 1 else sequences_df.iloc[:, 1].values

        print("Caching one-hot encodings...")
        self._cached_sequences = [one_hot_encode(seq) for seq in tqdm(self.sequences, desc="Encoding")]
        
        self._cached_features = None
        if self.use_engineered_features:
                print("Caching engineered features...")
            self._cached_features = [compute_engineered_features(seq) for seq in tqdm(self.sequences, desc="Features")]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
            sequence = self._cached_sequences[idx].copy()
        sequence_tensor = torch.from_numpy(sequence).float()

        if self.use_engineered_features and self._cached_features is not None:
            engineered = self._cached_features[idx].copy()
            engineered_tensor = torch.from_numpy(engineered).float()
            return sequence_tensor, engineered_tensor

        return sequence_tensor

# ==============================================================================
# INFERENCE
# ==============================================================================

def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and extract config + model state."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        model_state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" not in ckpt:
        config = None
        model_state = ckpt
    else:
        config = ckpt.get("config", None)
        model_state = ckpt.get("model_state_dict", ckpt)
    
    return config, model_state


def adapt_legacy_state_dict(state_dict):
    """
    Support older checkpoints whose classifier head was stored under `classifier.*`.
    Maps those parameters onto the current shared_mlp + class_head structure.
    """
    if "shared_mlp.0.weight" in state_dict or "class_head.weight" in state_dict:
        return state_dict

    legacy_linear1_w = state_dict.get("classifier.0.weight")
    if legacy_linear1_w is None:
        return state_dict

    print("Adapting legacy classifier weights to shared_mlp/class_head layout...")
    state_dict = state_dict.copy()
    # Shared MLP linear layer (Linear -> ReLU -> Dropout)
    state_dict["shared_mlp.0.weight"] = state_dict.pop("classifier.0.weight")
    state_dict["shared_mlp.0.bias"] = state_dict.pop("classifier.0.bias")
    # Final classification layer
    state_dict["class_head.weight"] = state_dict.pop("classifier.3.weight")
    state_dict["class_head.bias"] = state_dict.pop("classifier.3.bias")

    # Remove any leftover classifier entries
    keys_to_remove = [k for k in list(state_dict.keys()) if k.startswith("classifier.")]
    for k in keys_to_remove:
        state_dict.pop(k, None)

    return state_dict


def reconcile_config_with_state_dict(config, state_dict):
    """
    Make sure config flags align with what the checkpoint actually contains.
    """
    cfg = dict(config) if config else {}

    def has_prefix(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in state_dict.keys())

    # Engineered features head
    has_feature_mlp = has_prefix("feature_mlp.")
    if has_feature_mlp:
        cfg['use_engineered_features'] = True
    else:
        cfg['use_engineered_features'] = False

    if cfg.get('use_engineered_features', False):
        feat_in = state_dict.get("feature_mlp.0.weight")
        feat_out = state_dict.get("feature_mlp.3.weight")
        if feat_in is not None:
            cfg['engineered_feature_dim_in'] = feat_in.shape[1]
        if feat_out is not None:
            cfg['feature_dim'] = feat_out.shape[0]

    # Hierarchy toggles
    has_family_head = has_prefix("family_head.")
    has_subcluster_head = has_prefix("subcluster_head.")
    has_cond_heads = has_prefix("cond_heads.")

    if has_family_head or has_subcluster_head or has_cond_heads:
        cfg['use_hierarchy'] = True
        else:
        cfg['use_hierarchy'] = False

    if has_cond_heads:
        cfg['use_hierarchical_classifier'] = True
        else:
        cfg['use_hierarchical_classifier'] = False

    return cfg


def load_hierarchy_if_needed(config):
    """Load hierarchy metadata if model uses it."""
    if not config.get('use_hierarchy', False):
        return None
    
    # Try multiple paths
    paths_to_try = [
        config.get('hierarchy_path', ''),
        './phase8/label_hierarchy_v2.json',
        '../phase8/label_hierarchy_v2.json',
        '/content/drive/MyDrive/chromatin_phase8/label_hierarchy_v2.json'
    ]
    
    for path in paths_to_try:
        if path and os.path.exists(path):
            print(f"Loading hierarchy from: {path}")
            with open(path, 'r') as f:
                raw = json.load(f)
            
            # Build hierarchy metadata
            families = []
            max_subcluster = 0
            for k in sorted(raw.keys(), key=lambda x: int(x)):
                fam = raw[k]["family"]
                sub = int(raw[k]["subcluster"])
                families.append(fam)
                max_subcluster = max(max_subcluster, sub)
            
            uniq_families = sorted(set(families))
            family_to_id = {fam: i for i, fam in enumerate(uniq_families)}
            
            family_class_indices = [[] for _ in range(len(uniq_families))]
            for i in range(1, 19):
                entry = raw[str(i)]
                fam_id = family_to_id[entry["family"]]
                family_class_indices[fam_id].append(i - 1)
            
            for fam_id in range(len(family_class_indices)):
                family_class_indices[fam_id] = sorted(family_class_indices[fam_id])
            
            # Also need subcluster mapping for ConditionalSubclusterHead if used
            # But currently we only infer family_class_indices needed for HierarchicalClassifier
            # If the model has subcluster_head, we might need more metadata.
            # For now, let's assume we can instantiate with basic shapes or rely on stored config.
            
            # Note: Model init requires family_subcluster_indices for ConditionalSubclusterHead
            # We'll infer it similar to train.py if needed.

        return {
                "n_families": len(uniq_families),
                "n_subclusters": int(max_subcluster) + 1,
                "family_class_indices": family_class_indices,
                # "family_subcluster_indices": ... # Add if needed
            }
    
    print("Warning: Hierarchy file not found. Proceeding without hierarchy metadata (may fail if model requires it).")
    return None


def run_inference():
    """Run inference on test sequences and save predictions."""
    
    # Load checkpoint
    try:
        config, model_state = load_checkpoint(CONFIG['checkpoint_path'], device)
    except FileNotFoundError:
        print(f"Checkpoint not found at {CONFIG['checkpoint_path']}. Cannot run inference.")
        return
    
    # Adapt legacy naming before reconciling config
    model_state = adapt_legacy_state_dict(model_state)
    config = reconcile_config_with_state_dict(config, model_state)

    print("\nModel configuration (after reconciliation):")
    print(f"  n_classes: {config.get('n_classes', 18)}")
    print(f"  use_hierarchy: {config.get('use_hierarchy', False)}")
    print(f"  use_engineered_features: {config.get('use_engineered_features', False)}")
    print(f"  use_hierarchical_classifier: {config.get('use_hierarchical_classifier', False)}")
    
    # Load hierarchy if needed (post-reconciliation)
    hierarchy_meta = load_hierarchy_if_needed(config)
    
    # Create model
    # Note: If hierarchy_meta is missing but model requires it (e.g. cond_heads), this will crash or init wrong.
    # We try to use what we have.
    
    family_class_indices = hierarchy_meta["family_class_indices"] if hierarchy_meta else None
    
    # Important: If we lack hierarchy file but model expects cond_heads, we can't build the correct model structure
    # without knowing the family_class_indices sizes.
    # However, if we just want to load weights, maybe we can hack it? 
    # No, nn.ModuleList needs correct size.
    
    # Fallback: if family_class_indices is missing but required, we might guess from state_dict shapes?
    if config.get('use_hierarchical_classifier') and family_class_indices is None:
        print("CRITICAL: Missing hierarchy metadata for HierarchicalClassifier. Attempting to infer from state_dict...")
        # Infer from cond_heads weights
        # cond_heads.0.weight shape: (num_classes_in_fam, head_in)
        n_families = 0
        inferred_indices = []
        while True:
            key = f"hier_classifier.cond_heads.{n_families}.weight"
            if key in model_state:
                w = model_state[key]
                num_classes = w.shape[0]
                inferred_indices.append(list(range(num_classes))) # Dummy indices, just for count
                n_families += 1
                    else:
                break
        if n_families > 0:
            print(f"  Inferred {n_families} families from checkpoint.")
            family_class_indices = inferred_indices
            if hierarchy_meta is None: hierarchy_meta = {}
            hierarchy_meta["n_families"] = n_families
    
    model = ChromatinCNNAttention(
        n_classes=config.get('n_classes', 18),
        use_hierarchy=config.get('use_hierarchy', False),
        n_families=hierarchy_meta["n_families"] if hierarchy_meta else 0,
        n_subclusters=hierarchy_meta.get("n_subclusters", 0) if hierarchy_meta else 0,
        use_engineered_features=config.get('use_engineered_features', False),
        engineered_feature_dim_in=config.get('engineered_feature_dim_in', 5),
        feature_dim=config.get('feature_dim', 128),
        use_hierarchical_classifier=config.get('use_hierarchical_classifier', False),
            family_class_indices=family_class_indices,
        # family_subcluster_indices... (if needed for subclusters)
    )
    
    # Load weights
    model.load_state_dict(model_state, strict=False) # strict=False to be lenient with buffer/unused heads
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded successfully.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    test_dataset = TestDataset(
        CONFIG['test_sequences_csv'],
        use_engineered_features=config.get('use_engineered_features', False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=(device in ("cuda", "mps")),
    )
    
    print(f"\nRunning inference on {len(test_dataset):,} sequences...")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  RC TTA: {CONFIG['use_rc_tta']}")
    
    # Inference
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if config.get('use_engineered_features', False):
                sequences, engineered = batch
            sequences = sequences.to(device)
                engineered = engineered.to(device)
                else:
                sequences = batch
                sequences = sequences.to(device)
                engineered = None

            # Forward pass
                out = model(sequences, engineered=engineered)
            
            # Handle dict output
            if isinstance(out, dict):
                    logits = out["logits_class"]
                else:
                    logits = out
            
            # Optional RC TTA
            if CONFIG['use_rc_tta']:
                seq_rc = sequences.flip(dims=[1])[:, :, [3, 2, 1, 0]]
                out_rc = model(seq_rc, engineered=engineered)
                if isinstance(out_rc, dict):
                    logits_rc = out_rc["logits_class"]
        else:
                    logits_rc = out_rc
                logits = (logits + logits_rc) / 2
            
            # Get predictions (convert 0-17 to 1-18)
            preds = torch.argmax(logits, dim=1)
            all_predictions.extend((preds + 1).cpu().numpy())
    
    # Save predictions (just labels, no header as per competition format)
    print(f"\nSaving predictions to {CONFIG['output_csv']}")
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(CONFIG['output_csv'], index=False, header=False)
    
    print(f"✓ Saved {len(predictions_df):,} predictions")
    print(f"\nPrediction distribution:")
    print(pd.Series(all_predictions).value_counts().sort_index())
    
    print("\n✓ Inference complete!")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    run_inference()
