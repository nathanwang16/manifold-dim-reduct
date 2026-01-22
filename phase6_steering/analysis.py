# -*- coding: utf-8 -*-
"""phase6_analysis_improved.py

Adapted from phase6_colab.ipynb for the improved ChromatinCNNAttention architecture.
Produces full analysis reports including calibration, steering, and figures.
"""

# ==============================================================================
# 1. Environment & Imports
# ==============================================================================
import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# Check device
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("Using CPU")

# Mount Google Drive (if running in Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Colab, skipping Drive mount")

# Set paths (Modified for 'improved' checkpoints)
DATA_PATH = '/content/drive/MyDrive/chromatin_data'
CHECKPOINT_PATH = '/content/drive/MyDrive/chromatin_model_checkpoints_improved'
OUTPUT_PATH = '/content/drive/MyDrive/chromatin_phase6_improved'

print(f"Data path: {DATA_PATH}")
print(f"Checkpoint path: {CHECKPOINT_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# Ensure output dirs exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_PATH}/cache").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_PATH}/results").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_PATH}/results/figures").mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Logger & Utils
# ==============================================================================
class ColabLogger:
    def __init__(self, name: str): self.name = name
    def info(self, msg: str): print(f"[INFO] {self.name}: {msg}")
    def warning(self, msg: str): print(f"[WARN] {self.name}: {msg}")
    def error(self, msg: str): print(f"[ERROR] {self.name}: {msg}")

logger = ColabLogger("Phase6")

@contextmanager
def LogTimer(logger, description: str):
    start = time.time()
    logger.info(f"Starting: {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {description} ({elapsed:.2f}s)")

def compute_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax probabilities with temperature scaling."""
    scaled = logits / temperature
    # Stability trick: shift by max
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_logits = np.exp(scaled)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vectors / norms

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    norm1 = np.linalg.norm(v1) + eps
    norm2 = np.linalg.norm(v2) + eps
    return np.dot(v1, v2) / (norm1 * norm2)

class MetricsTracker:
    def __init__(self, logger_instance=None):
        self.metrics: Dict[str, List[float]] = {}
        self.logger = logger_instance or logger

    def log(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, values in self.metrics.items():
            if values:
                arr = np.array(values)
                summary[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'last': float(arr[-1]),
                    'count': len(values),
                }
        return summary

# ==============================================================================
# 3. Model Architecture (ChromatinCNNAttention)
# ==============================================================================

try:
    from phase3_model.model_defs import ChromatinCNNAttention, ResidualBlock
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from phase3_model.model_defs import ChromatinCNNAttention, ResidualBlock

def load_model(checkpoint_path, config, device):
    logger.info(f"Loading model from {checkpoint_path}")
    # Initialize model with hardcoded architecture matching the improved version
    # 'config' is accepted but not strictly used to instantiate architecture 
    # to ensure we match the specific structure of 'ChromatinCNNAttention'
    model = ChromatinCNNAttention(n_classes=18)
    
    # Checkpoint loading
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise e
        
    model = model.to(device)
    model.eval()
    return model

# ==============================================================================
# 4. Data Loading
# ==============================================================================

def one_hot_encode(sequence: str) -> np.ndarray:
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((200, 4), dtype=np.float32)
    for i, base in enumerate(sequence):
        if base in base_to_idx:
            one_hot[i, base_to_idx[base]] = 1.0
        else:
            one_hot[i, :] = 0.25
    return one_hot

class ChromatinDataset(Dataset):
    def __init__(self, sequences_file, labels_file=None, rc_augment=False):
        self.sequences_file = sequences_file
        self.labels_file = labels_file
        self.rc_augment = rc_augment
        
        print(f"Loading sequences from {sequences_file}")
        sequences_df = pd.read_csv(sequences_file, header=None)
        self.sequences = sequences_df.iloc[:, 1].values if sequences_df.shape[1] > 1 else sequences_df[0].values
        
        self.labels = None
        if labels_file:
            print(f"Loading labels from {labels_file}")
            labels_df = pd.read_csv(labels_file, header=None)
            self.labels = labels_df.iloc[:, 1].values if labels_df.shape[1] > 1 else labels_df[0].values
            self.labels = self.labels - 1 

        self._cached_sequences = [one_hot_encode(seq) for seq in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        one_hot = self._cached_sequences[idx].copy()
        if self.rc_augment and np.random.random() < 0.5:
            one_hot = one_hot[::-1, [3, 2, 1, 0]].copy()
            
        sequence_tensor = torch.from_numpy(one_hot)
        label_tensor = torch.tensor(self.labels[idx] if self.labels is not None else 0, dtype=torch.long)
        return sequence_tensor, label_tensor

# ==============================================================================
# 5. Analysis Utilities (Activation Cache, Steering, Temperature)
# ==============================================================================

class ActivationCache:
    def __init__(self, model, device, cache_dir):
        self.model = model
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._activations = None
        self._labels = None
        self._predictions = None
        self._logits = None

    def extract_activations(self, data_loader, layer_name='bottleneck'):
        logger.info(f"Extracting activations from layer: {layer_name}")
        all_acts, all_lbls, all_preds, all_logits = [], [], [], []
        
        with torch.no_grad():
            for sequences, labels in tqdm(data_loader, desc="Extracting"):
                sequences = sequences.to(self.device)
                logits, activations_dict = self.model(sequences, return_activations=True)
                
                acts = activations_dict[layer_name]
                # If acts are 3D (B, C, L), pool them. If 2D (B, C), keep as is.
                if acts.dim() == 3:
                    acts = torch.mean(acts, dim=2)
                
                all_acts.append(acts.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_lbls.append(labels.numpy())
                
        self._activations = np.concatenate(all_acts, axis=0)
        self._labels = np.concatenate(all_lbls, axis=0)
        self._predictions = np.concatenate(all_preds, axis=0)
        self._logits = np.concatenate(all_logits, axis=0)
        
        return self._activations, self._labels, self._predictions, self._logits

    def save_cache(self, name):
        path = self.cache_dir / f"{name}.npz"
        np.savez_compressed(
            path, 
            activations=self._activations, 
            labels=self._labels, 
            predictions=self._predictions,
            logits=self._logits
        )
        logger.info(f"Saved cache to {path}")

    def load_cache(self, name):
        path = self.cache_dir / f"{name}.npz"
        data = np.load(path)
        self._activations = data['activations']
        self._labels = data['labels']
        self._predictions = data['predictions']
        self._logits = data['logits']
        return self._activations, self._labels, self._predictions, self._logits

    def cache_exists(self, name):
        return (self.cache_dir / f"{name}.npz").exists()

class SteeringVectorComputer:
    def __init__(self, n_classes=18, n_features=256):
        self.n_classes = n_classes
        self.label_centroids = np.zeros((n_classes, n_features))
        
    def compute_centroids(self, activations, labels):
        logger.info("Computing label centroids...")
        for label in range(self.n_classes):
            mask = labels == label
            if np.sum(mask) > 0:
                self.label_centroids[label] = np.mean(activations[mask], axis=0)
                
    def get_steering_vector(self, src_label, tgt_label, normalize=True):
        vec = self.label_centroids[tgt_label] - self.label_centroids[src_label]
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-8: vec /= norm
        return vec
        
    def save(self, path):
        np.savez_compressed(path, label_centroids=self.label_centroids)
        logger.info(f"Saved steering vectors to {path}")
        
    @classmethod
    def load(cls, path):
        data = np.load(path)
        instance = cls()
        instance.label_centroids = data['label_centroids']
        instance.n_classes = instance.label_centroids.shape[0]
        return instance

class TemperatureScaler:
    def __init__(self, n_classes=18):
        self.n_classes = n_classes
        self.temperature = 1.0

    def fit(self, logits, labels):
        logger.info("Fitting temperature scaler...")
        
        def nll(t):
            t = max(t, 0.01) # Avoid division by zero
            scaled_logits = logits / t
            probs = compute_softmax(scaled_logits)
            # NLL
            rows = np.arange(len(labels))
            p = probs[rows, labels]
            return -np.mean(np.log(np.clip(p, 1e-10, 1.0)))

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = float(res.x)
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
        return self.temperature

    def calibrate(self, logits):
        return compute_softmax(logits, temperature=self.temperature)

    def compute_reliability(self, logits, labels, n_bins=15):
        probs = compute_softmax(logits, temperature=self.temperature)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_confs, bin_accs, bin_counts = [], [], []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            count = np.sum(in_bin)
            bin_counts.append(count)
            if count > 0:
                bin_confs.append(np.mean(confidences[in_bin]))
                bin_accs.append(np.mean(accuracies[in_bin]))
            else:
                bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
                bin_accs.append(0.0)
                
        # ECE
        total = np.sum(bin_counts)
        ece = np.sum(np.abs(np.array(bin_accs) - np.array(bin_confs)) * np.array(bin_counts)) / total if total > 0 else 0
        
        return {
            'bin_confidences': bin_confs,
            'bin_accuracies': bin_accs,
            'bin_counts': bin_counts,
            'ece': ece
        }
        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'temperature': self.temperature}, f)

# ==============================================================================
# 6. Steering Inference Engines
# ==============================================================================

class SteeringInferenceEngine:
    def __init__(self, model, steering_computer, device):
        self.model = model
        self.steering_computer = steering_computer
        self.device = device
        
    def apply_steering(self, sequences, target_label, source_label=None, alpha=0.5):
        sequences = sequences.to(self.device)
        with torch.no_grad():
            orig_logits, activations = self.model(sequences, return_activations=True)
            bottleneck = activations['bottleneck'] # (B, 256)
            orig_preds = torch.argmax(orig_logits, dim=1)
            
            # Create steering vectors
            steered_bottleneck = bottleneck.clone()
            batch_size = sequences.shape[0]
            
            for i in range(batch_size):
                src = source_label if source_label is not None else orig_preds[i].item()
                vec = self.steering_computer.get_steering_vector(src, target_label, normalize=False)
                vec_t = torch.from_numpy(vec).float().to(self.device)
                steered_bottleneck[i] += alpha * vec_t
                
            # Forward from bottleneck (classifier part of ChromatinCNNAttention)
            steered_logits = self.model.classifier(steered_bottleneck)
            
        return steered_logits, orig_logits, orig_preds

class ContrastiveSteeringEngine:
    def __init__(self, steering_engine):
        self.steering_engine = steering_engine
        self.model = steering_engine.model
        self.steering_computer = steering_engine.steering_computer
        self.device = steering_engine.device
        self.confused_pairs = []

    def compute_confusion_matrix(self, preds, labels):
        cm = sklearn_confusion_matrix(labels, preds, labels=list(range(18)))
        # Row normalize
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = cm / np.maximum(row_sums, 1)
        
        # Identify pairs
        pairs = []
        for i in range(18):
            for j in range(18):
                if i != j and norm_cm[i, j] > 0.05: # Threshold
                    pairs.append({'label_i': int(i), 'label_j': int(j), 'score': float(norm_cm[i, j])})
        
        pairs.sort(key=lambda x: x['score'], reverse=True)
        self.confused_pairs = pairs[:10]
        return self.confused_pairs

    def apply_contrastive_correction(self, sequences, alpha=0.5, conf_thresh=0.6):
        # Simplified batch correction:
        # 1. Get original predictions
        # 2. If confident, keep.
        # 3. If not confident and predicted class is in a confused pair, try steering to the other class in pair.
        #    If steering increases confidence significantly, flip.
        
        # For this script, we'll just run a simple evaluation loop
        sequences = sequences.to(self.device)
        with torch.no_grad():
            logits, activations = self.model(sequences, return_activations=True)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            
            corrected_preds = preds.clone()
            corrections_mask = torch.zeros_like(preds, dtype=torch.bool)
            
            bottleneck = activations['bottleneck']
            
            # Iterate only over low confidence samples
            uncertain_indices = torch.where(confs < conf_thresh)[0]
            
            for idx in uncertain_indices:
                p = preds[idx].item()
                # Check if p is part of a confused pair
                # We simply check the top confused pair involving p
                candidate = None
                for pair in self.confused_pairs:
                    if pair['label_i'] == p: candidate = pair['label_j']
                    elif pair['label_j'] == p: candidate = pair['label_i']
                    
                    if candidate is not None:
                        # Try steering towards candidate
                        vec = self.steering_computer.get_steering_vector(p, candidate, normalize=False)
                        steered_b = bottleneck[idx] + alpha * torch.from_numpy(vec).float().to(self.device)
                        steered_l = self.model.classifier(steered_b.unsqueeze(0))
                        steered_p = F.softmax(steered_l, dim=1)
                        
                        # If candidate probability is now high, swap
                        if steered_p[0, candidate] > steered_p[0, p] and steered_p[0, candidate] > conf_thresh:
                            corrected_preds[idx] = candidate
                            corrections_mask[idx] = True
                        break # Only try top pair
                        
        return corrected_preds, corrections_mask

# ==============================================================================
# 7. Alignment Evaluator
# ==============================================================================

class AlignmentEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = TemperatureScaler()
        
    def compute_rc_consistency(self, data_loader):
        logger.info("Computing RC Consistency...")
        total, consistent = 0, 0
        with torch.no_grad():
            for seqs, _ in tqdm(data_loader):
                seqs = seqs.to(self.device)
                # Original
                preds_orig = torch.argmax(self.model(seqs), dim=1)
                # RC
                seqs_rc = seqs.flip(dims=[1])[:, :, [3, 2, 1, 0]]
                preds_rc = torch.argmax(self.model(seqs_rc), dim=1)
                
                consistent += (preds_orig == preds_rc).sum().item()
                total += seqs.size(0)
        return consistent / total

    def test_monotonicity(self, base_seq, motif_seq, pos, expected_label):
        # base_seq: (1, 200, 4)
        target_seq = base_seq.clone()
        motif_len = motif_seq.shape[1]
        target_seq[0, pos:pos+motif_len, :] = motif_seq[0]
        
        alphas = np.linspace(0, 1, 10)
        confs = []
        
        with torch.no_grad():
            for alpha in alphas:
                interp = (1 - alpha) * base_seq + alpha * target_seq
                interp = interp.to(self.device)
                logits = self.model(interp)
                probs = F.softmax(logits, dim=1)
                confs.append(probs[0, expected_label].item())
                
        # Score
        increases = 0
        for i in range(len(confs)-1):
            if confs[i+1] >= confs[i] - 1e-4:
                increases += 1
        return increases / (len(confs)-1)

    def generate_report(self, val_loader, activations, labels, logits, output_dir):
        report = {}
        
        # 1. RC Consistency
        report['rc_consistency'] = {'consistency_rate': self.compute_rc_consistency(val_loader)}
        
        # 2. Calibration
        self.scaler.fit(logits, labels)
        self.scaler.save(Path(output_dir) / 'temperature.json')
        
        rel_before = self.scaler.compute_reliability(logits, labels)
        # Hack to compute "after" without re-running everything, just re-use fitted scaler internally
        # But compute_reliability uses self.temperature, so we need to set it to 1.0 to get "before" stats correctly?
        # Actually compute_reliability uses self.temperature.
        # So to get "before", we need a temp=1.0 scaler.
        scaler_identity = TemperatureScaler(); scaler_identity.temperature = 1.0
        rel_before = scaler_identity.compute_reliability(logits, labels)
        rel_after = self.scaler.compute_reliability(logits, labels)
        
        report['calibration'] = {
            'ece_before': float(rel_before['ece']),
            'ece_after': float(rel_after['ece']),
            'temperature': float(self.scaler.temperature)
        }
        
        # Plot Reliability Diagram
        self.plot_reliability(rel_before, rel_after, Path(output_dir) / 'results/figures/reliability_diagram.png')
        
        return report

    def plot_reliability(self, rel_before, rel_after, path):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, data, title in [(axes[0], rel_before, 'Before Calibration'), (axes[1], rel_after, 'After Calibration')]:
            accs = data['bin_accuracies']
            confs = data['bin_confidences']
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            ax.plot(confs, accs, 'o-')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(f"{title}\nECE = {data['ece']:.4f}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

def _string_to_onehot(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    onehot = torch.zeros(1, len(seq), 4)
    for i, base in enumerate(seq.upper()):
        if base in mapping: onehot[0, i, mapping[base]] = 1.0
        else: onehot[0, i, :] = 0.25
    return onehot

# ==============================================================================
# 8. Main Execution Flow
# ==============================================================================

def main():
    # 8.1 Setup
    best_model_path = os.path.join(CHECKPOINT_PATH, 'best_model_improved.pt')
    if not os.path.exists(best_model_path):
        print("Checkpoint not found! Please run training first.")
        return

    # Configuration Dictionary (Added as requested)
    config = {
        'n_classes': 18,
        'model_type': 'ChromatinCNNAttention'
    }
    
    model = load_model(best_model_path, config, DEVICE)
    tracker = MetricsTracker(logger)

    # 8.2 Load Data
    val_dataset = ChromatinDataset(
        f"{DATA_PATH}/val_sequences.csv", 
        f"{DATA_PATH}/val_labels.csv"
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Also load train for steering vectors
    train_dataset = ChromatinDataset(
        f"{DATA_PATH}/train_sequences.csv", 
        f"{DATA_PATH}/train_labels.csv"
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 8.3 Extract Activations
    cache = ActivationCache(model, DEVICE, f"{OUTPUT_PATH}/cache")
    
    # Train Activations (for Steering)
    if cache.cache_exists('train_activations'):
        train_acts, train_lbls, _, _ = cache.load_cache('train_activations')
    else:
        train_acts, train_lbls, _, _ = cache.extract_activations(train_loader)
        cache.save_cache('train_activations')
        
    # Val Activations (for Evaluation)
    if cache.cache_exists('val_activations'):
        val_acts, val_lbls, val_preds, val_logits = cache.load_cache('val_activations')
    else:
        val_acts, val_lbls, val_preds, val_logits = cache.extract_activations(val_loader)
        cache.save_cache('val_activations')

    val_acc = np.mean(val_preds == val_lbls)
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    tracker.log('val_accuracy', val_acc)

    # 8.4 Compute Steering Vectors
    steering_computer = SteeringVectorComputer(n_classes=18, n_features=train_acts.shape[1])
    steering_computer.compute_centroids(train_acts, train_lbls)
    steering_computer.save(f"{OUTPUT_PATH}/cache/steering_vectors.npz")

    # 8.5 Alignment Evaluation (RC, Calibration, Monotonicity)
    evaluator = AlignmentEvaluator(model, DEVICE)
    
    # Run monotonic tests
    print("Running Monotonicity Tests...")
    base_seq = torch.zeros(1, 200, 4) + 0.25
    motif_seq = _string_to_onehot("TATAAA")
    mono_score = evaluator.test_monotonicity(base_seq, motif_seq, 30, 0)
    tracker.log('monotonicity_tata', mono_score)
    
    # Generate Report
    align_report = evaluator.generate_report(val_loader, val_acts, val_lbls, val_logits, OUTPUT_PATH)
    with open(f"{OUTPUT_PATH}/results/alignment_report.json", 'w') as f:
        json.dump(align_report, f, indent=2)
        
    tracker.log('rc_consistency', align_report['rc_consistency']['consistency_rate'])
    tracker.log('ece_after', align_report['calibration']['ece_after'])

    # 8.6 Contrastive Steering Evaluation
    steering_engine = SteeringInferenceEngine(model, steering_computer, DEVICE)
    contrastive_engine = ContrastiveSteeringEngine(steering_engine)
    
    # Compute confusion
    confused_pairs = contrastive_engine.compute_confusion_matrix(val_preds, val_lbls)
    
    # Eval improvement
    logger.info("Evaluating Contrastive Steering...")
    val_seqs_np = np.stack(val_dataset._cached_sequences)
    corr_preds, corr_mask = contrastive_engine.apply_contrastive_correction(
        torch.from_numpy(val_seqs_np).float(), # Use cached full dataset tensor for simplicity
        alpha=0.5
    )
    
    corr_acc = np.mean(corr_preds.cpu().numpy() == val_lbls)
    improvement = corr_acc - val_acc
    logger.info(f"Corrected Accuracy: {corr_acc:.4f} (Improvement: {improvement:+.4f})")
    
    confusion_report = {
        'confused_pairs': confused_pairs,
        'original_accuracy': float(val_acc),
        'corrected_accuracy': float(corr_acc),
        'improvement': float(improvement)
    }
    with open(f"{OUTPUT_PATH}/results/confusion_analysis.json", 'w') as f:
        json.dump(confusion_report, f, indent=2)
        
    tracker.log('contrastive_improvement', improvement)

    # 8.7 Final Summary
    summary = tracker.summarize()
    with open(f"{OUTPUT_PATH}/results/phase6_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to {OUTPUT_PATH}")
    print("Files generated:")
    print(f" - {OUTPUT_PATH}/results/alignment_report.json")
    print(f" - {OUTPUT_PATH}/results/confusion_analysis.json")
    print(f" - {OUTPUT_PATH}/results/phase6_summary.json")
    print(f" - {OUTPUT_PATH}/results/temperature.json")
    print(f" - {OUTPUT_PATH}/results/figures/reliability_diagram.png")

if __name__ == "__main__":
    main()
