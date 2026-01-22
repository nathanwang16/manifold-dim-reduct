# -*- coding: utf-8 -*-
"""phase7_diagnostics.py

Diagnostic suite for ChromatinCNNAttention model.
Focuses on Causal Interpretation using Saliency Maps and Confusion Analysis.
"""

# ==============================================================================
# 1. Environment & Imports
# ==============================================================================
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Check device
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("Using CPU")

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Colab, skipping Drive mount")

# Paths
DATA_PATH = '/content/drive/MyDrive/chromatin_data'
CHECKPOINT_PATH = '/content/drive/MyDrive/chromatin_model_checkpoints_improved'
OUTPUT_PATH = '/content/drive/MyDrive/chromatin_phase7_diagnostics'

# Ensure output dirs
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_PATH}/saliency_maps").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_PATH}/confusion_examples").mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Biological Label Mapping (Hypothetical Standard 18-State)
# ==============================================================================
# Based on common Roadmap Epigenomics 18-state models. 
# Adjust if your specific dataset uses a different mapping.
STATE_LABELS = {
    0: "1_TssA (Active TSS)",
    1: "2_TssFlnk (Flanking TSS)",
    2: "3_TssFlnkU (Flanking TSS Up)",
    3: "4_TssFlnkD (Flanking TSS Down)",
    4: "5_Tx (Strong Transcription)",
    5: "6_TxWk (Weak Transcription)",
    6: "7_EnhG1 (Genic Enhancer 1)",
    7: "8_EnhG2 (Genic Enhancer 2)",
    8: "9_EnhA1 (Active Enhancer 1)",
    9: "10_EnhA2 (Active Enhancer 2)",
    10: "11_EnhWk (Weak Enhancer)",
    11: "12_ZNF/Rpts (ZNF genes & repeats)",
    12: "13_Het (Heterochromatin)",
    13: "14_TssBiv (Bivalent/Poised TSS)",
    14: "15_EnhBiv (Bivalent Enhancer)",
    15: "16_ReprPC (Repressed PolyComb)",
    16: "17_ReprPCWk (Weak Repressed PolyComb)",
    17: "18_Quies (Quiescent/Low)"
}

def get_label_name(idx):
    return STATE_LABELS.get(idx, f"Class_{idx}")

# ==============================================================================
# 3. Model Architecture (Must match trained model)
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class ChromatinCNNAttention(nn.Module):
    def __init__(self, n_classes=18, input_len=200):
        super().__init__()
        self.n_classes = n_classes
        
        # 1. Stem
        self.stem_conv = nn.Conv1d(4, 128, kernel_size=19, padding='same', bias=True)
        self.stem_bn = nn.BatchNorm1d(128)
        
        # 2. Residual Tower
        self.res_block1 = ResidualBlock(128, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2) 
        
        self.conv_expand = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.res_block2 = ResidualBlock(256, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2) 
        
        # 3. Attention
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.norm_attn = nn.LayerNorm(256)
        
        # 4. Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x, return_activations=False):
        if x.shape[1] == 200:
            x = x.transpose(1, 2)
            
        activations = {}
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.res_block1(x)
        x = self.pool1(x)
        x = F.relu(self.conv_expand(x))
        x = self.res_block2(x)
        x = self.pool2(x)
        
        x_perm = x.permute(0, 2, 1) 
        attn_out, _ = self.attention(x_perm, x_perm, x_perm)
        x_perm = self.norm_attn(x_perm + attn_out)
        x = x_perm.permute(0, 2, 1)
        
        x_pooled = self.global_pool(x).squeeze(-1)
        activations['bottleneck'] = x_pooled 
        logits = self.classifier(x_pooled)
        
        if return_activations:
            return logits, activations
        return logits

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}")
    model = ChromatinCNNAttention(n_classes=18)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
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
    def __init__(self, sequences_file, labels_file=None):
        self.sequences_file = sequences_file
        self.labels_file = labels_file
        
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
        sequence_tensor = torch.from_numpy(one_hot)
        label_tensor = torch.tensor(self.labels[idx] if self.labels is not None else 0, dtype=torch.long)
        return sequence_tensor, label_tensor, idx

# ==============================================================================
# 5. Saliency Analysis Engine
# ==============================================================================
class SaliencyExplorer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_saliency(self, sequence_tensor, target_class=None):
        """
        Computes Input x Gradient saliency.
        sequence_tensor: (1, 200, 4) or (1, 4, 200)
        target_class: int (optional). If None, uses predicted class.
        """
        self.model.eval()
        
        # Ensure input requires grad
        if not sequence_tensor.requires_grad:
            sequence_tensor.requires_grad_()
            
        # Forward pass
        logits = self.model(sequence_tensor)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass on target score
        score = logits[0, target_class]
        score.backward()
        
        # Get gradients (Input x Gradient)
        # Gradient shape: same as input (1, 200, 4)
        gradients = sequence_tensor.grad.data.cpu().numpy()[0]
        
        # Saliency magnitude: Sum of absolute gradients across channels at each position
        # Or simply L2 norm at each position
        # Using L2 norm per position to see "where" the model looks
        saliency = np.linalg.norm(gradients, axis=1) # Shape (200,)
        
        return saliency, gradients, target_class, logits.detach().cpu().numpy()[0]

    def plot_saliency(self, sequence_onehot, saliency, gradients, title, save_path):
        """
        Plots saliency map with sequence overlay.
        """
        seq_len = saliency.shape[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
        
        # 1. Saliency Magnitude Bar Chart
        ax1.bar(range(seq_len), saliency, color='black', alpha=0.7)
        ax1.set_title(f"Saliency Magnitude: {title}")
        ax1.set_ylabel("Gradient Magnitude")
        
        # 2. Sequence Logo-like visualization (approximate)
        # We define base colors
        colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
        bases = ['A', 'C', 'G', 'T']
        
        # Reconstruct sequence string from onehot
        seq_indices = np.argmax(sequence_onehot, axis=1)
        seq_str = [bases[i] for i in seq_indices]
        
        # Plot gradients per channel
        # gradients shape (200, 4)
        for i, base in enumerate(bases):
            ax2.plot(gradients[:, i], label=base, color=colors[base], alpha=0.6)
            
        # Highlight actual base positions
        # This is crowded for 200bp, so we just label high saliency spots?
        # Let's simple plot the dominant gradient channel
        
        ax2.set_ylabel("Gradient (Signed)")
        ax2.set_xlabel("Position (bp)")
        ax2.legend(loc='upper right')
        ax2.set_title("Gradients per Nucleotide Channel")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def aggregate_saliency_profile(self, saliency_list):
        """Computes mean saliency profile across multiple examples."""
        stack = np.stack(saliency_list) # (N, 200)
        mean_profile = np.mean(stack, axis=0)
        std_profile = np.std(stack, axis=0)
        return mean_profile, std_profile

# ==============================================================================
# 6. Main Diagnostic Routine
# ==============================================================================
def main():
    # 1. Load Resources
    best_model_path = os.path.join(CHECKPOINT_PATH, 'best_model_improved.pt')
    if not os.path.exists(best_model_path):
        print(f"Checkpoint not found at {best_model_path}")
        return
        
    model = load_model(best_model_path, DEVICE)
    
    val_dataset = ChromatinDataset(
        f"{DATA_PATH}/val_sequences.csv", 
        f"{DATA_PATH}/val_labels.csv"
    )
    # Use batch size 1 for granular analysis, or larger for search
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    explorer = SaliencyExplorer(model, DEVICE)
    
    # 2. Identify Confusion Targets (Focus on 0 vs 13)
    TARGET_PAIRS = [(0, 13)] # Add more if needed from confusion analysis
    
    confusion_log = []
    class_0_saliencies = []
    class_13_saliencies = []
    
    print("Scanning validation set for Correct and Confused examples...")
    
    # Limits to prevent overflow
    max_examples = 50
    counts = {'0_correct': 0, '13_correct': 0, '0_as_13': 0, '13_as_0': 0}
    
    for seq_tensor, label_tensor, idx in tqdm(val_loader):
        label = label_tensor.item()
        seq_tensor = seq_tensor.to(DEVICE).float().requires_grad_()
        
        # Forward
        logits = model(seq_tensor)
        pred = logits.argmax(dim=1).item()
        
        # Store for aggregation
        if label == 0 and pred == 0 and counts['0_correct'] < max_examples:
            sal, _, _, _ = explorer.compute_saliency(seq_tensor, target_class=0)
            class_0_saliencies.append(sal)
            counts['0_correct'] += 1
            
        elif label == 13 and pred == 13 and counts['13_correct'] < max_examples:
            sal, _, _, _ = explorer.compute_saliency(seq_tensor, target_class=13)
            class_13_saliencies.append(sal)
            counts['13_correct'] += 1
            
        # Analyze Confusion: True 0, Pred 13
        elif label == 0 and pred == 13 and counts['0_as_13'] < 5:
            print(f"\n[Found Confusion] Sample {idx.item()}: True {get_label_name(0)} -> Pred {get_label_name(13)}")
            
            # Compute Saliency for the WRONG prediction (Why did it think 13?)
            sal_wrong, grads_wrong, _, _ = explorer.compute_saliency(seq_tensor, target_class=13)
            explorer.plot_saliency(
                seq_tensor.detach().cpu().numpy()[0], 
                sal_wrong, grads_wrong, 
                f"Why Pred {get_label_name(13)}? (True: {get_label_name(0)})",
                f"{OUTPUT_PATH}/confusion_examples/sample_{idx.item()}_why_pred_13.png"
            )
            
            # Compute Saliency for the TRUE class (What did it miss?)
            # Need to re-forward since gradients were cleared
            seq_tensor.grad = None
            logits = model(seq_tensor) # Re-forward
            sal_true, grads_true, _, _ = explorer.compute_saliency(seq_tensor, target_class=0)
            explorer.plot_saliency(
                seq_tensor.detach().cpu().numpy()[0], 
                sal_true, grads_true, 
                f"Evidence for True {get_label_name(0)} (Missed)",
                f"{OUTPUT_PATH}/confusion_examples/sample_{idx.item()}_evidence_true_0.png"
            )
            
            counts['0_as_13'] += 1
            
    # 3. Aggregate Analysis
    print("\nGenerating Aggregate Saliency Profiles...")
    
    # Class 0 Profile
    if class_0_saliencies:
        mean_0, _ = explorer.aggregate_saliency_profile(class_0_saliencies)
        plt.figure(figsize=(12, 4))
        plt.plot(mean_0, color='blue', label='Mean Saliency')
        plt.title(f"Average Importance Profile: {get_label_name(0)} (N={len(class_0_saliencies)})")
        plt.xlabel("Position")
        plt.ylabel("Gradient Magnitude")
        plt.savefig(f"{OUTPUT_PATH}/saliency_maps/avg_profile_class_0.png")
        plt.close()
        
    # Class 13 Profile
    if class_13_saliencies:
        mean_13, _ = explorer.aggregate_saliency_profile(class_13_saliencies)
        plt.figure(figsize=(12, 4))
        plt.plot(mean_13, color='red', label='Mean Saliency')
        plt.title(f"Average Importance Profile: {get_label_name(13)} (N={len(class_13_saliencies)})")
        plt.xlabel("Position")
        plt.ylabel("Gradient Magnitude")
        plt.savefig(f"{OUTPUT_PATH}/saliency_maps/avg_profile_class_13.png")
        plt.close()

    print(f"\nDiagnostics Complete. Results saved to {OUTPUT_PATH}")
    print("Check 'confusion_examples' folder for individual failure cases.")
    print("Check 'saliency_maps' for global class profiles.")

if __name__ == "__main__":
    main()






