"""
Chromatin State Label Identification Pipeline

This module identifies which competition labels (1-18) correspond to which 
known biological chromatin states based on sequence characteristics.

Reference: Roadmap Epigenomics 18-state model (Kundaje et al., Nature 2015)

Known States and Their Sequence Signatures:
==========================================
1. TssA (Active TSS)        - CpG islands, high GC (>60%), TATA/Inr motifs
2. TssFlnk (Flanking TSS)   - High GC, near CpG regions
3. TssFlnkU (TSS upstream)  - Promoter-adjacent signatures
4. TssFlnkD (TSS downstream)- Transition into gene body
5. Tx (Strong transcription)- Gene body composition, moderate GC
6. TxWk (Weak transcription)- Similar to Tx but weaker signal
7. EnhG1 (Genic enhancer 1) - H3K4me1 signature regions, specific TF motifs
8. EnhG2 (Genic enhancer 2) - Similar to EnhG1
9. EnhA1 (Active enhancer 1)- AP-1, ETS family motifs
10. EnhA2 (Active enhancer 2)- Similar TF binding sites
11. EnhWk (Weak enhancer)   - Weaker enhancer signatures
12. ZnfRpts (ZNF genes/repeats)- KRAB-ZNF motifs, repetitive
13. Het (Heterochromatin)   - AT-rich (>65%), satellite repeats
14. TssBiv (Bivalent TSS)   - CpG islands + developmental gene motifs
15. EnhBiv (Bivalent enhancer)- Poised enhancer signatures
16. ReprPC (Repressed Polycomb)- CpG-rich, developmental targets
17. ReprPCWk (Weak Polycomb) - Similar but weaker
18. Quies (Quiescent)       - Background composition (~40% GC)
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple
import re
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import linear_sum_assignment


# =============================================================================
# PART 1: Sequence Feature Extraction
# =============================================================================

def compute_gc_content(sequence: str) -> float:
    """Calculate GC content of a sequence."""
    gc = sum(1 for base in sequence.upper() if base in 'GC')
    return gc / len(sequence) if len(sequence) > 0 else 0

def compute_cpg_ratio(sequence: str) -> float:
    """
    Calculate CpG observed/expected ratio.
    CpG O/E = (CpG count * length) / (C count * G count)
    Values > 0.6 indicate CpG islands (promoter-associated).
    """
    seq = sequence.upper()
    cpg = seq.count('CG')
    c = seq.count('C')
    g = seq.count('G')
    if c == 0 or g == 0:
        return 0
    expected = (c * g) / len(seq)
    return cpg / expected if expected > 0 else 0

def compute_kmer_frequencies(sequence: str, k: int = 4) -> Dict[str, float]:
    """Compute normalized k-mer frequencies."""
    seq = sequence.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    return {kmer: count/total for kmer, count in counts.items()}

def compute_dinucleotide_frequencies(sequence: str) -> Dict[str, float]:
    """Compute all 16 dinucleotide frequencies."""
    return compute_kmer_frequencies(sequence, k=2)

def count_homopolymer_runs(sequence: str, min_length: int = 4) -> Dict[str, int]:
    """Count runs of repeated nucleotides (indicator of low complexity)."""
    seq = sequence.upper()
    runs = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for base in 'ACGT':
        pattern = base * min_length
        runs[base] = seq.count(pattern)
    return runs

def compute_repeat_density(sequence: str) -> float:
    """
    Estimate repetitive element density.
    Simple repeats are characteristic of heterochromatin.
    """
    seq = sequence.upper()
    repeat_patterns = [
        r'(AT){4,}',      # AT repeats
        r'(TA){4,}',      # TA repeats  
        r'(CA){4,}',      # CA repeats
        r'(TG){4,}',      # TG repeats
        r'([ACGT])\1{5,}', # Homopolymers 6+
        r'(AAT){3,}',     # Satellite-like
        r'(ATT){3,}',
    ]
    repeat_bases = 0
    for pattern in repeat_patterns:
        for match in re.finditer(pattern, seq):
            repeat_bases += match.end() - match.start()
    return min(repeat_bases / len(seq), 1.0)  # Cap at 1.0

def scan_known_motifs(sequence: str) -> Dict[str, int]:
    """
    Scan for known regulatory motifs associated with chromatin states.
    Returns count of each motif type found.
    """
    seq = sequence.upper()
    
    motif_patterns = {
        # Promoter-associated
        'TATA_box': r'TATA[AT]A[AT]',
        'CAAT_box': r'GG[CT]CAATCT',
        'GC_box': r'GGGCGG',
        'Inr': r'[CT][CT]A[ACGT][AT][CT][CT]',  # Initiator
        
        # Enhancer-associated TF motifs
        'AP1': r'TGA[CG]TCA',      # AP-1 (Jun/Fos)
        'ETS': r'[AC]GGA[AT]G',    # ETS family
        'GATA': r'[AT]GATA[AG]',   # GATA factors
        'CEBP': r'T[TG]NNGNAA[TG]', # C/EBP
        
        # Polycomb-associated
        'CpG_dense': r'CGCG',
        
        # CTCF/Insulator
        'CTCF_core': r'CC[AG]C[CG]AGGGGGC', # CTCF consensus
        
        # ZNF-associated
        'KRAB_ZNF': r'TGCAG',  # Common in KRAB-ZNF genes
    }
    
    results = {}
    for name, pattern in motif_patterns.items():
        matches = re.findall(pattern, seq)
        results[name] = len(matches)
    
    return results

def extract_all_features(sequence: str) -> Dict[str, float]:
    """Extract comprehensive feature set from a sequence."""
    features = {}
    
    # Basic composition
    features['gc_content'] = compute_gc_content(sequence)
    features['cpg_ratio'] = compute_cpg_ratio(sequence)
    features['repeat_density'] = compute_repeat_density(sequence)
    
    # AT content (inverse of GC, but explicit for clarity)
    features['at_content'] = 1 - features['gc_content']
    
    # Dinucleotide frequencies
    dinucs = compute_dinucleotide_frequencies(sequence)
    for dinuc, freq in dinucs.items():
        features[f'dinuc_{dinuc}'] = freq
    
    # CpG specifically (important for promoters)
    features['cpg_freq'] = dinucs.get('CG', 0)
    
    # Homopolymer runs
    runs = count_homopolymer_runs(sequence)
    features['poly_A'] = runs['A']
    features['poly_T'] = runs['T']
    features['total_homopolymers'] = sum(runs.values())
    
    # Motif counts
    motifs = scan_known_motifs(sequence)
    for motif_name, count in motifs.items():
        features[f'motif_{motif_name}'] = count
    
    return features


# =============================================================================
# PART 2: Expected Profiles for Known Chromatin States
# =============================================================================

def get_expected_state_profiles() -> Dict[str, Dict[str, float]]:
    """
    Define expected feature profiles for each known chromatin state.
    Based on literature characterization of epigenomic states.
    
    Values are relative scores (0-1) indicating expected feature levels.
    """
    profiles = {
        # State 1: Active TSS - highest CpG, high GC
        'TssA': {
            'gc_content': 0.65, 'cpg_ratio': 0.85, 'cpg_freq': 0.08,
            'repeat_density': 0.05, 'motif_TATA_box': 0.3, 'motif_GC_box': 0.6,
            'motif_Inr': 0.4, 'at_content': 0.35, 'total_homopolymers': 0.1
        },
        # State 2: Flanking Active TSS
        'TssFlnk': {
            'gc_content': 0.55, 'cpg_ratio': 0.6, 'cpg_freq': 0.05,
            'repeat_density': 0.1, 'motif_TATA_box': 0.2, 'motif_GC_box': 0.4,
            'at_content': 0.45, 'total_homopolymers': 0.15
        },
        # States 3-4: TSS upstream/downstream
        'TssFlnkU': {
            'gc_content': 0.52, 'cpg_ratio': 0.55, 'cpg_freq': 0.04,
            'repeat_density': 0.1, 'at_content': 0.48, 'total_homopolymers': 0.15
        },
        'TssFlnkD': {
            'gc_content': 0.50, 'cpg_ratio': 0.45, 'cpg_freq': 0.035,
            'repeat_density': 0.12, 'at_content': 0.50, 'total_homopolymers': 0.18
        },
        # State 5: Strong transcription (gene body)
        'Tx': {
            'gc_content': 0.45, 'cpg_ratio': 0.35, 'cpg_freq': 0.025,
            'repeat_density': 0.15, 'at_content': 0.55, 'total_homopolymers': 0.2
        },
        # State 6: Weak transcription
        'TxWk': {
            'gc_content': 0.42, 'cpg_ratio': 0.30, 'cpg_freq': 0.02,
            'repeat_density': 0.18, 'at_content': 0.58, 'total_homopolymers': 0.25
        },
        # States 7-8: Genic enhancers
        'EnhG1': {
            'gc_content': 0.48, 'cpg_ratio': 0.40, 'cpg_freq': 0.03,
            'repeat_density': 0.12, 'motif_AP1': 0.3, 'motif_ETS': 0.3,
            'at_content': 0.52, 'total_homopolymers': 0.18
        },
        'EnhG2': {
            'gc_content': 0.46, 'cpg_ratio': 0.38, 'cpg_freq': 0.028,
            'repeat_density': 0.14, 'motif_AP1': 0.25, 'motif_ETS': 0.28,
            'at_content': 0.54, 'total_homopolymers': 0.2
        },
        # States 9-10: Active enhancers
        'EnhA1': {
            'gc_content': 0.47, 'cpg_ratio': 0.42, 'cpg_freq': 0.032,
            'repeat_density': 0.10, 'motif_AP1': 0.4, 'motif_ETS': 0.35,
            'motif_GATA': 0.25, 'at_content': 0.53, 'total_homopolymers': 0.15
        },
        'EnhA2': {
            'gc_content': 0.45, 'cpg_ratio': 0.38, 'cpg_freq': 0.028,
            'repeat_density': 0.12, 'motif_AP1': 0.35, 'motif_ETS': 0.30,
            'at_content': 0.55, 'total_homopolymers': 0.18
        },
        # State 11: Weak enhancer
        'EnhWk': {
            'gc_content': 0.42, 'cpg_ratio': 0.32, 'cpg_freq': 0.022,
            'repeat_density': 0.18, 'motif_AP1': 0.15, 'motif_ETS': 0.15,
            'at_content': 0.58, 'total_homopolymers': 0.22
        },
        # State 12: ZNF genes & repeats
        'ZnfRpts': {
            'gc_content': 0.52, 'cpg_ratio': 0.45, 'cpg_freq': 0.035,
            'repeat_density': 0.35, 'motif_KRAB_ZNF': 0.5,
            'at_content': 0.48, 'total_homopolymers': 0.3
        },
        # State 13: Heterochromatin - AT-rich, repeat-rich
        'Het': {
            'gc_content': 0.35, 'cpg_ratio': 0.20, 'cpg_freq': 0.01,
            'repeat_density': 0.45, 'at_content': 0.65,
            'total_homopolymers': 0.5, 'poly_A': 0.4, 'poly_T': 0.4
        },
        # State 14: Bivalent/Poised TSS
        'TssBiv': {
            'gc_content': 0.58, 'cpg_ratio': 0.70, 'cpg_freq': 0.06,
            'repeat_density': 0.08, 'motif_CpG_dense': 0.5,
            'at_content': 0.42, 'total_homopolymers': 0.12
        },
        # State 15: Bivalent enhancer
        'EnhBiv': {
            'gc_content': 0.48, 'cpg_ratio': 0.45, 'cpg_freq': 0.035,
            'repeat_density': 0.15, 'motif_AP1': 0.2, 'motif_ETS': 0.2,
            'at_content': 0.52, 'total_homopolymers': 0.2
        },
        # State 16: Repressed Polycomb
        'ReprPC': {
            'gc_content': 0.50, 'cpg_ratio': 0.55, 'cpg_freq': 0.04,
            'repeat_density': 0.12, 'motif_CpG_dense': 0.35,
            'at_content': 0.50, 'total_homopolymers': 0.18
        },
        # State 17: Weak repressed Polycomb
        'ReprPCWk': {
            'gc_content': 0.45, 'cpg_ratio': 0.40, 'cpg_freq': 0.028,
            'repeat_density': 0.18, 'at_content': 0.55, 'total_homopolymers': 0.22
        },
        # State 18: Quiescent/Low signal - background genome composition
        'Quies': {
            'gc_content': 0.40, 'cpg_ratio': 0.25, 'cpg_freq': 0.015,
            'repeat_density': 0.25, 'at_content': 0.60, 'total_homopolymers': 0.35
        }
    }
    return profiles


# =============================================================================
# PART 3: Label Identification Algorithm
# =============================================================================

def compute_label_profiles(sequences: List[str], labels: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Compute average feature profiles for each competition label.
    """
    label_features = {label: [] for label in range(1, 19)}
    
    print("Extracting features from sequences...")
    for i, (seq, label) in enumerate(zip(sequences, labels)):
        if i % 10000 == 0:
            print(f"  Processing sequence {i}/{len(sequences)}")
        features = extract_all_features(seq)
        label_features[label].append(features)
    
    # Compute mean profile for each label
    label_profiles = {}
    for label in range(1, 19):
        if label_features[label]:
            all_keys = set()
            for f in label_features[label]:
                all_keys.update(f.keys())
            
            profile = {}
            for key in all_keys:
                values = [f.get(key, 0) for f in label_features[label]]
                profile[key] = np.mean(values)
            label_profiles[label] = profile
    
    return label_profiles

def match_labels_to_states(
    label_profiles: Dict[int, Dict[str, float]],
    expected_profiles: Dict[str, Dict[str, float]],
    key_features: List[str] = None
) -> Tuple[Dict[int, str], np.ndarray]:
    """
    Match competition labels to known chromatin states using feature similarity.
    
    Returns:
        mapping: Dict mapping label number to state name
        similarity_matrix: (18 x 18) matrix of label vs state similarities
    """
    if key_features is None:
        key_features = [
            'gc_content', 'cpg_ratio', 'cpg_freq', 'repeat_density',
            'at_content', 'total_homopolymers'
        ]
    
    state_names = list(expected_profiles.keys())
    
    # Build feature matrices
    n_labels = 18
    n_states = 18
    n_features = len(key_features)
    
    label_matrix = np.zeros((n_labels, n_features))
    state_matrix = np.zeros((n_states, n_features))
    
    for i, label in enumerate(range(1, 19)):
        if label in label_profiles:
            for j, feat in enumerate(key_features):
                label_matrix[i, j] = label_profiles[label].get(feat, 0)
    
    for i, state in enumerate(state_names):
        for j, feat in enumerate(key_features):
            state_matrix[i, j] = expected_profiles[state].get(feat, 0)
    
    # Normalize
    label_matrix = (label_matrix - label_matrix.mean(axis=0)) / (label_matrix.std(axis=0) + 1e-8)
    state_matrix = (state_matrix - state_matrix.mean(axis=0)) / (state_matrix.std(axis=0) + 1e-8)
    
    # Compute distance / similarity
    distances = cdist(label_matrix, state_matrix, metric='euclidean')
    similarity = -distances

    # Optimal one-to-one assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(distances)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(r) + 1] = state_names[int(c)]

    return mapping, similarity


def generate_identification_report(
    label_profiles: Dict[int, Dict[str, float]],
    mapping: Dict[int, str],
    similarity: np.ndarray
) -> str:
    """Generate a detailed report of the label identification results."""
    expected = get_expected_state_profiles()
    state_names = list(expected.keys())
    
    report = []
    report.append("=" * 80)
    report.append("CHROMATIN STATE LABEL IDENTIFICATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary mapping
    report.append("LABEL → STATE MAPPING")
    report.append("-" * 40)
    for label in range(1, 19):
        state = mapping.get(label, "Unknown")
        conf = similarity[label-1].max()
        report.append(f"  Label {label:2d} → {state:12s} (confidence: {conf:.3f})")
    
    report.append("")
    report.append("KEY FEATURE COMPARISON")
    report.append("-" * 80)
    
    # Feature comparison
    key_feats = ['gc_content', 'cpg_ratio', 'repeat_density', 'at_content']
    header = f"{'Label':>6} {'State':>12} | " + " | ".join(f"{f:>12}" for f in key_feats)
    report.append(header)
    report.append("-" * len(header))
    
    for label in range(1, 19):
        state = mapping.get(label, "?")
        profile = label_profiles.get(label, {})
        values = [f"{profile.get(f, 0):.4f}" for f in key_feats]
        report.append(f"{label:>6} {state:>12} | " + " | ".join(f"{v:>12}" for v in values))
    
    report.append("")
    report.append("CONFIDENCE ANALYSIS")
    report.append("-" * 40)
    
    # Identify uncertain mappings
    for label in range(1, 19):
        sims = similarity[label-1]
        sorted_idx = np.argsort(-sims)
        top1, top2 = state_names[sorted_idx[0]], state_names[sorted_idx[1]]
        conf1, conf2 = sims[sorted_idx[0]], sims[sorted_idx[1]]
        margin = conf1 - conf2
        
        if margin < 0.5:
            report.append(f"  ⚠️  Label {label}: Low confidence margin ({margin:.3f})")
            report.append(f"       Best: {top1} ({conf1:.3f}) vs {top2} ({conf2:.3f})")
    
    return "\n".join(report)


# =============================================================================
# PART 4: Main Execution
# =============================================================================

def run_label_identification(
    train_sequences_path: str,
    train_labels_path: str,
    output_dir: str = "label_identification_output"
) -> Dict[int, str]:
    """
    Main function to run the complete label identification pipeline.
    
    Args:
        train_sequences_path: Path to trainsequences.csv
        train_labels_path: Path to trainlabels.csv
        output_dir: Directory to save outputs
        
    Returns:
        mapping: Dictionary mapping label numbers to state names
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading training data...")
    with open(train_sequences_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    with open(train_labels_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Loaded {len(sequences)} sequences with {len(set(labels))} unique labels")
    
    # Sample for faster computation (use all for final analysis)
    sample_size = min(50000, len(sequences))
    np.random.seed(42)
    indices = np.random.choice(len(sequences), sample_size, replace=False)
    sampled_seqs = [sequences[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    
    # Compute profiles
    print("\nComputing feature profiles per label...")
    label_profiles = compute_label_profiles(sampled_seqs, sampled_labels)
    
    # Get expected profiles
    expected = get_expected_state_profiles()
    
    # Match labels to states
    print("\nMatching labels to known chromatin states...")
    mapping, similarity = match_labels_to_states(label_profiles, expected)
    
    # Generate report
    report = generate_identification_report(label_profiles, mapping, similarity)
    print("\n" + report)
    
    # Save outputs
    with open(f"{output_dir}/identification_report.txt", 'w') as f:
        f.write(report)
    
    # Save mapping as JSON
    import json
    with open(f"{output_dir}/label_state_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)
    
    # Save profiles for further analysis
    profiles_df = pd.DataFrame(label_profiles).T
    profiles_df.to_csv(f"{output_dir}/label_feature_profiles.csv")
    
    # Save similarity matrix
    np.save(f"{output_dir}/similarity_matrix.npy", similarity)
    
    print(f"\nResults saved to {output_dir}/")
    
    return mapping


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Identify chromatin state labels")
    parser.add_argument("--sequences", default="data/trainsequences.csv")
    parser.add_argument("--labels", default="data/trainlabels.csv")
    parser.add_argument("--output", default="label_identification_output")
    args = parser.parse_args()
    
    mapping = run_label_identification(args.sequences, args.labels, args.output)
    
    print("\n" + "=" * 50)
    print("FINAL MAPPING:")
    print("=" * 50)
    for label, state in sorted(mapping.items()):
        print(f"  Label {label:2d} = {state}")
