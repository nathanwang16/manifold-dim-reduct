"""
Visualization and validation for chromatin state label identification.

This script creates visualizations to validate the label→state mapping
and understand the structure of the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple
import json

# Import from main module
from label_identification_plan import (
    extract_all_features, 
    get_expected_state_profiles,
    compute_gc_content,
    compute_cpg_ratio,
    compute_repeat_density
)


def quick_feature_profile(sequences: List[str], labels: List[int]) -> pd.DataFrame:
    """
    Compute quick feature profiles for visualization.
    Uses only the most discriminative features.
    """
    print("Computing quick feature profiles...")
    
    data = []
    for i, (seq, label) in enumerate(zip(sequences, labels)):
        if i % 5000 == 0:
            print(f"  {i}/{len(sequences)}")
        
        data.append({
            'label': label,
            'gc_content': compute_gc_content(seq),
            'cpg_ratio': compute_cpg_ratio(seq),
            'repeat_density': compute_repeat_density(seq),
            'at_content': 1 - compute_gc_content(seq),
        })
    
    return pd.DataFrame(data)


def visualize_label_distributions(df: pd.DataFrame, output_dir: str = "."):
    """Create violin plots showing feature distributions per label."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    features = ['gc_content', 'cpg_ratio', 'repeat_density', 'at_content']
    titles = ['GC Content', 'CpG O/E Ratio', 'Repeat Density', 'AT Content']
    
    for ax, feat, title in zip(axes.flatten(), features, titles):
        # Prepare data for violin plot
        positions = list(range(1, 19))
        data = [df[df['label'] == l][feat].values for l in positions]
        
        parts = ax.violinplot(data, positions=positions, showmeans=True)
        
        # Color by expected state type
        # States 1-4: Promoter (red), 5-6: Tx (green), 7-11: Enhancer (orange)
        # 12: ZNF (purple), 13: Het (gray), 14-15: Bivalent (cyan)
        # 16-17: Polycomb (blue), 18: Quies (black)
        
        ax.set_xlabel('Label')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Distribution by Label')
        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=150)
    plt.close()
    print(f"Saved feature distributions to {output_dir}/feature_distributions.png")


def visualize_label_scatter(df: pd.DataFrame, output_dir: str = "."):
    """Create scatter plots showing label separation in feature space."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: GC content vs CpG ratio
    ax = axes[0]
    for label in range(1, 19):
        subset = df[df['label'] == label]
        ax.scatter(subset['gc_content'], subset['cpg_ratio'], 
                  alpha=0.3, s=10, label=f'{label}')
    ax.set_xlabel('GC Content')
    ax.set_ylabel('CpG O/E Ratio')
    ax.set_title('GC Content vs CpG Ratio')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    # Plot 2: GC content vs repeat density
    ax = axes[1]
    for label in range(1, 19):
        subset = df[df['label'] == label]
        ax.scatter(subset['gc_content'], subset['repeat_density'],
                  alpha=0.3, s=10, label=f'{label}')
    ax.set_xlabel('GC Content')
    ax.set_ylabel('Repeat Density')
    ax.set_title('GC Content vs Repeat Density')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_scatter.png", dpi=150)
    plt.close()
    print(f"Saved label scatter to {output_dir}/label_scatter.png")


def visualize_label_centroids(df: pd.DataFrame, output_dir: str = "."):
    """Plot label centroids with expected state annotations."""
    
    # Compute centroids
    centroids = df.groupby('label')[['gc_content', 'cpg_ratio', 'repeat_density']].mean()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot centroids
    scatter = ax.scatter(centroids['gc_content'], centroids['cpg_ratio'],
                        c=centroids.index, cmap='tab20', s=200, edgecolors='black')
    
    # Add labels
    for label in centroids.index:
        ax.annotate(str(label), 
                   (centroids.loc[label, 'gc_content'], 
                    centroids.loc[label, 'cpg_ratio']),
                   fontsize=10, ha='center', va='center')
    
    # Add expected regions
    ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='Promoter GC threshold')
    ax.axhline(y=0.6, color='blue', linestyle='--', alpha=0.5, label='CpG island threshold')
    ax.axvline(x=0.38, color='gray', linestyle='--', alpha=0.5, label='Het GC threshold')
    
    ax.set_xlabel('Mean GC Content', fontsize=12)
    ax.set_ylabel('Mean CpG O/E Ratio', fontsize=12)
    ax.set_title('Label Centroids in Feature Space', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_centroids.png", dpi=150)
    plt.close()
    print(f"Saved label centroids to {output_dir}/label_centroids.png")


def identify_states_heuristic(df: pd.DataFrame) -> Dict[int, str]:
    """
    Use simple heuristics to identify likely state mappings.
    This is a quick sanity check before running full identification.
    """
    centroids = df.groupby('label')[['gc_content', 'cpg_ratio', 'repeat_density']].mean()
    
    # Heuristic identifications
    likely_states = {}
    
    # TssA: Highest CpG ratio + high GC
    tssa_candidates = centroids[(centroids['cpg_ratio'] > 0.5) & 
                                (centroids['gc_content'] > 0.55)]
    if len(tssa_candidates) > 0:
        tssa_label = tssa_candidates['cpg_ratio'].idxmax()
        likely_states[tssa_label] = "TssA (Active Promoter)"
    
    # Het: Lowest GC + low CpG
    het_candidates = centroids[centroids['gc_content'] < 0.4]
    if len(het_candidates) > 0:
        het_label = het_candidates['gc_content'].idxmin()
        likely_states[het_label] = "Het (Heterochromatin)"
    
    # ZnfRpts or Het: Highest repeat density
    high_repeat = centroids['repeat_density'].idxmax()
    if high_repeat not in likely_states:
        likely_states[high_repeat] = "ZnfRpts or Het (high repeats)"
    
    # Quies: Near genome average (GC ~40%, moderate CpG)
    quies_candidates = centroids[
        (centroids['gc_content'].between(0.38, 0.44)) &
        (centroids['cpg_ratio'] < 0.35)
    ]
    if len(quies_candidates) > 0:
        # Pick one closest to (0.41, 0.25)
        distances = np.sqrt((quies_candidates['gc_content'] - 0.41)**2 + 
                           (quies_candidates['cpg_ratio'] - 0.25)**2)
        quies_label = distances.idxmin()
        if quies_label not in likely_states:
            likely_states[quies_label] = "Quies (Quiescent)"
    
    return likely_states


def print_ranking_table(df: pd.DataFrame):
    """Print labels ranked by key features."""
    centroids = df.groupby('label')[['gc_content', 'cpg_ratio', 'repeat_density']].mean()
    
    print("\n" + "=" * 70)
    print("LABELS RANKED BY KEY FEATURES")
    print("=" * 70)
    
    print("\nBy GC Content (highest first - likely promoters):")
    gc_ranked = centroids['gc_content'].sort_values(ascending=False)
    for i, (label, gc) in enumerate(gc_ranked.items()):
        state_hint = ""
        if i < 5: state_hint = " ← likely promoter/TSS"
        if i >= 15: state_hint = " ← likely Het/Quies"
        print(f"  {i+1:2d}. Label {label:2d}: GC={gc:.4f}{state_hint}")
    
    print("\nBy CpG O/E Ratio (highest first - likely CpG islands):")
    cpg_ranked = centroids['cpg_ratio'].sort_values(ascending=False)
    for i, (label, cpg) in enumerate(cpg_ranked.items()):
        state_hint = ""
        if i < 3: state_hint = " ← likely TssA/TssBiv"
        if cpg > 0.6: state_hint = " ← CpG island"
        print(f"  {i+1:2d}. Label {label:2d}: CpG_ratio={cpg:.4f}{state_hint}")
    
    print("\nBy Repeat Density (highest first - likely Het/ZnfRpts):")
    rep_ranked = centroids['repeat_density'].sort_values(ascending=False)
    for i, (label, rep) in enumerate(rep_ranked.items()):
        state_hint = ""
        if i < 3: state_hint = " ← likely Het or ZnfRpts"
        print(f"  {i+1:2d}. Label {label:2d}: repeat_density={rep:.4f}{state_hint}")


def run_visualization_pipeline(
    sequences_path: str,
    labels_path: str,
    output_dir: str = "visualization_output",
    sample_size: int = 30000
):
    """Run the complete visualization pipeline."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    with open(sequences_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    with open(labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Sample for faster visualization
    np.random.seed(42)
    indices = np.random.choice(len(sequences), min(sample_size, len(sequences)), replace=False)
    sampled_seqs = [sequences[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    
    # Compute features
    df = quick_feature_profile(sampled_seqs, sampled_labels)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_label_distributions(df, output_dir)
    visualize_label_scatter(df, output_dir)
    visualize_label_centroids(df, output_dir)
    
    # Print ranking table
    print_ranking_table(df)
    
    # Heuristic identification
    print("\n" + "=" * 70)
    print("HEURISTIC STATE IDENTIFICATIONS")
    print("=" * 70)
    likely_states = identify_states_heuristic(df)
    for label, state in sorted(likely_states.items()):
        print(f"  Label {label:2d} → {state}")
    
    # Save dataframe
    df.to_csv(f"{output_dir}/feature_data.csv", index=False)
    print(f"\nFeature data saved to {output_dir}/feature_data.csv")
    
    return df, likely_states


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="data/trainsequences.csv")
    parser.add_argument("--labels", default="data/trainlabels.csv")
    parser.add_argument("--output", default="visualization_output")
    parser.add_argument("--sample", type=int, default=30000)
    args = parser.parse_args()
    
    df, likely_states = run_visualization_pipeline(
        args.sequences, args.labels, args.output, args.sample
    )
