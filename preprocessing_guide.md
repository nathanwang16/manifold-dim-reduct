# Roadmap Epigenomics Dataset Preprocessing Guide -- Second stage of research with expanded volumn of dataset
### Chromatin State Prediction — hg19 → hg38 Pipeline

**Target:** 18-state ChromHMM annotations → balanced (sequence, label) pairs in hg38  
**Input:** Roadmap BED segmentation files (hg19), UCSC liftOver chain, hg38 FASTA  
**Output:** `sequences.csv` / `labels.csv` pairs per epigenome, merged training corpus

---

## Table of Contents

1. [Environment & Dependencies](#1-environment--dependencies)
2. [File Inventory & Validation](#2-file-inventory--validation)
3. [BED File Normalization](#3-bed-file-normalization)
4. [LiftOver: Coordinate Conversion](#4-liftover-coordinate-conversion)
5. [Post-LiftOver Filtering](#5-post-liftover-filtering)
6. [Sequence Extraction](#6-sequence-extraction)
7. [Sequence-Level Filtering](#7-sequence-level-filtering)
8. [State Label Normalization](#8-state-label-normalization)
9. [Cross-Epigenome Deduplication](#9-cross-epigenome-deduplication)
10. [Dataset Balancing](#10-dataset-balancing)
11. [Train / Val / Test Split](#11-train--val--test-split)
12. [Quality Checks & Sanity Metrics](#12-quality-checks--sanity-metrics)
13. [Final Schema](#13-final-schema)
14. [Epigenome Selection Reference](#14-epigenome-selection-reference)

---

## 1. Environment & Dependencies

### Required External Tools

| Tool | Purpose | Version Requirement |
|------|---------|-------------------|
| `liftOver` | UCSC coordinate conversion binary | Any; download from UCSC |
| `samtools` | FASTA indexing | ≥1.9 |
| `bedtools` | Sequence extraction from FASTA | ≥2.27 |
| `bgzip` / `tabix` | BED decompression and indexing | Part of htslib |

### Required Reference Files

| File | Source | Notes |
|------|--------|-------|
| `hg19ToHg38.over.chain.gz` | UCSC liftOver chains | Shared across all epigenomes |
| `hg38.fa.gz` | UCSC hg38 bigZips | Primary sequence reference |
| `hg38.fa.fai` | Generated from `samtools faidx` | Required for fast random access |
| `hg19.fa.gz` | UCSC hg19 bigZips | Optional; for boundary validation |

### Python Libraries

| Library | Purpose |
|---------|---------|
| `pyfaidx` | Python-native FASTA random access; alternative to bedtools |
| `pandas` | BED parsing, metadata joins, label mapping |
| `numpy` | Array operations during balancing and statistics |
| `scipy` | Hungarian algorithm for label validation (if re-running Phase 8) |
| `tqdm` | Progress tracking across 127 epigenomes |

### Directory Layout

Adopt this layout before starting. Many intermediate files share similar names; a flat directory will become unmanageable at scale.

```
project_root/
├── raw/
│   ├── beds/           # Downloaded .bed.gz files, one per epigenome
│   └── metadata/       # EID metadata table from Roadmap portal
├── reference/
│   ├── hg19.fa(.gz/.fai)
│   ├── hg38.fa(.gz/.fai)
│   └── hg19ToHg38.over.chain.gz
├── intermediate/
│   ├── normalized/     # Post-normalization BEDs (before liftover)
│   ├── lifted/         # Post-liftover BEDs (hg38 coordinates)
│   ├── unmapped/       # LiftOver rejection logs
│   └── sequences/      # Per-epigenome FASTA extractions
├── processed/
│   ├── per_epigenome/  # Final (sequences.csv, labels.csv) pairs
│   └── merged/         # Cross-epigenome aggregated dataset
└── logs/
    └── qc/             # Per-step QC reports
```

---

## 2. File Inventory & Validation

### Epigenome BED Files

The Roadmap portal hosts 127 segmentation files. Not all 127 are equal quality. Before downloading everything, cross-reference the metadata table against these criteria:

**Inclusion criteria (all must be true):**
- Epigenome marked as using **observed** data (not imputed) for the 6 core marks: H3K4me3, H3K4me1, H3K27me3, H3K36me3, H3K27ac, H3K9me3
- Epigenome has a clear tissue/cell-type group assignment
- BED file size > 50MB (very small files indicate incomplete annotations)

**Recommended starting subset (~30 epigenomes):**
Select one representative per tissue group from the Roadmap metadata. Priority groups: ESC (E003), iPSC (E020), blood T-cell (E034), blood B-cell (E032), brain prefrontal cortex (E069), liver (E066), lung (E096), skeletal muscle (E108), mammary epithelial (E119), fibroblast (E055), heart (E095), kidney (E086). This covers major tissue diversity without the full computational burden of 127 epigenomes.

### File Integrity Check

After download, verify each `.bed.gz` file:
- Decompress and count lines. Expect **~15 million lines per epigenome** (human genome / 200bp ≈ 15M bins)
- Files significantly smaller than 14M lines likely have assembly gaps handled differently — flag for inspection
- Spot-check chromosome coverage: every canonical chromosome (chr1-chr22, chrX, chrY) should be present

### Metadata Table Columns to Retain

From the Roadmap metadata table, keep these columns for downstream joins:

| Column | Use |
|--------|-----|
| `EID` | Primary key for all joins |
| `Epigenome name` | Human-readable label for logging |
| `GROUP` | Tissue group for stratified sampling |
| `TYPE` | Primary / derived cell type |
| `Anatomy` | Organ system |
| `MARKS` | Which marks were observed vs. imputed |

---

## 3. BED File Normalization

Before liftover, every BED file must pass through a normalization step to ensure consistent format. Inconsistencies here will silently propagate through the pipeline.

### Expected Raw Format

```
chr1    0       200     18_Quies
chr1    200     400     7_Enh
chr1    400     600     7_Enh
...
```

Columns: chromosome, start (0-based), end (exclusive), state name string.

### Normalization Checks (in order)

**3.1 Chromosome prefix consistency**  
All chromosomes must use the `chr` prefix. Roadmap files do use this convention, but verify with a `head` inspection. If any file uses bare integers (`1`, `2`, ...) add the prefix before liftover — UCSC liftOver requires the `chr` prefix.

**3.2 Interval uniformity**  
Every interval must satisfy `end - start == 200` exactly. Compute `end - start` across all rows and assert the distribution is a single spike at 200. Any deviation indicates a malformed or truncated file. Drop any rows where this fails — do not attempt to fix them.

**3.3 State name vocabulary**  
Extract the unique set of state names from each file. The valid 18-state vocabulary is:

```
1_TssA, 2_TssFlnk, 3_TssFlnkU, 4_TssFlnkD, 5_Tx, 6_TxWk,
7_EnhG1, 8_EnhG2, 9_EnhA1, 10_EnhA2, 11_EnhWk, 12_ZNF/Rpts,
13_Het, 14_TssBiv, 15_EnhBiv, 16_ReprPC, 17_ReprPCWk, 18_Quies
```

Any name outside this set indicates either a different model (15-state, 25-state) or a naming convention change. Do not mix models.

**3.4 Chromosome whitelist**  
Remove rows where chromosome is not in the canonical set: `{chr1, ..., chr22, chrX, chrY}`. This removes unplaced contigs before liftover — no point lifting coordinates that will be discarded anyway.

**3.5 Sort order**  
LiftOver performs better on sorted input. Sort each BED by chromosome (lexicographic on the `chr` string is fine here), then by start position numerically. Use a stable sort.

### Output of Normalization

One normalized `.bed` file per epigenome in `intermediate/normalized/`. Log the number of rows dropped at each check step per epigenome. If any epigenome drops >5% of rows during normalization, investigate before proceeding.

---

## 4. LiftOver: Coordinate Conversion

### Conceptual Overview

LiftOver maps each interval's coordinates from hg19 to hg38 by consulting the chain file, which encodes syntenic blocks between assemblies. A 200bp bin either:
- **Maps cleanly:** All 200bp land in a single contiguous hg38 block → keep
- **Maps partially:** The interval spans a chain break → reject
- **Maps to multiple locations:** Rare; reject
- **Does not map:** Coordinates are in hg19-only regions (e.g., some repeat regions) → reject and log

### Key LiftOver Parameters

**`-minMatch`:** The minimum fraction of the input interval that must map to the output. Set this to `0.95` or higher. Since ChromHMM bins have no partial biological meaning — a bin is either a 200bp unit or nothing — there is no value in keeping partially mapped bins. A bin that maps 190bp is not a meaningful 190bp training example; it's a corrupt 200bp example.

**Do not use `-multiple`:** This flag allows one-to-many mappings (one hg19 interval maps to multiple hg38 locations). This happens in duplicated regions of hg38. If you allow it, you'll get duplicate sequences with the same label, which inflates your dataset artificially. Keep the default behavior (reject multiply-mapping intervals).

### Running LiftOver

Run liftOver separately for each epigenome, producing:
- A **mapped output BED** (hg38 coordinates, same state column retained)
- An **unmapped output BED** (rejected intervals with rejection reason as comment lines)

### Interpreting the Unmapped File

The unmapped file prefixes each rejected interval with a comment line explaining why. The common reasons and what they imply:

| Reason | Implication |
|--------|------------|
| `#Deleted in new` | Region exists in hg19 but was removed in hg38 assembly; expected loss |
| `#Split in new` | Interval spans a chain break; correct to reject |
| `#Partially deleted` | Sub-threshold mapping; correct to reject at minMatch=0.95 |
| `#Duplicated in new` | Maps to multiple locations; reject (see `-multiple` note above) |

**Acceptable loss rate:** 5–10% of bins rejected is normal. Loss rates above 15% in a specific chromosome should be investigated — this can indicate chromosome-level assembly differences (e.g., chrY is substantially restructured between hg19 and hg38).

### Post-LiftOver Coordinate Validation

After liftover, the mapped file's intervals are in hg38 coordinates but may not all be exactly 200bp. The liftOver chain can occasionally produce slight length changes at chain boundaries. Verify:
- Compute `end - start` for all mapped intervals
- Keep only rows where the result is exactly 200
- Log how many are dropped at this step (should be <0.1%)

---

## 5. Post-LiftOver Filtering

### Chromosome Whitelist (Re-apply)

LiftOver can introduce coordinates on alternative contigs even if your input was clean. Re-apply the canonical chromosome whitelist filter after liftover. This is not redundant — it catches cases where a canonical hg19 region maps to an alternative contig in hg38.

### Coordinate Bounds Check

For each chromosome, verify that all interval coordinates fall within the known hg38 chromosome lengths. Coordinates beyond the chromosome end are invalid and will cause errors during sequence extraction. Use the hg38 chromosome length table (available in the `.fai` index file as the second column).

### Overlap Deduplication Within an Epigenome

LiftOver can occasionally produce two output intervals that overlap, if two originally adjacent non-overlapping hg19 bins map to overlapping hg38 coordinates. This is rare but must be handled. Within each epigenome's lifted BED:
- Sort by chromosome and start position
- Identify any pair of consecutive rows on the same chromosome where `row[i].start < row[i-1].end`
- Drop both rows in the pair (neither can be trusted as a clean 200bp unit)

---

## 6. Sequence Extraction

### FASTA Index Requirement

Before any extraction, the hg38 FASTA must be indexed with `samtools faidx`. The resulting `.fai` file enables O(1) random access to any position in the genome. Without it, extracting millions of intervals requires streaming the entire FASTA repeatedly — computationally infeasible.

The `.fai` format stores, per chromosome: name, total bases, byte offset of first base, bases per line, bytes per line. `pyfaidx` reads this index directly in Python; `bedtools getfasta` uses it implicitly.

### Extraction Approaches

**Option A — bedtools getfasta:**  
Pass the lifted BED file directly. Use the `-name` flag to embed the interval coordinates in the FASTA header, which lets you trace each sequence back to its genomic position later. Output is a multi-entry FASTA file.

**Option B — pyfaidx (Python):**  
More controllable programmatically. Load the genome once, then query each interval by coordinate. Handles strand logic explicitly if needed. Better for pipelines that need to attach metadata to each extraction.

### Strand Handling

ChromHMM segmentations are strand-agnostic — a state annotation applies to both strands simultaneously. For sequence extraction, always extract from the **positive strand** (the reference strand as stored in the FASTA). Do not flip to reverse complement at this stage. Reverse complement handling is a training augmentation concern, not a data pipeline concern.

### Output Format

Extract sequences as plain text, one sequence per line, with a parallel label file of the same length. This mirrors the competition format exactly and makes merging across epigenomes straightforward. Do not store as FASTA with headers in the final format — headers add parsing overhead and are not needed once the label file maintains the correspondence.

Intermediately, you may want to retain the BED coordinates alongside each sequence during the extraction phase (before generating the final CSVs), so you can trace any sequence back to its genomic origin for QC purposes.

---

## 7. Sequence-Level Filtering

### N-Filtering (Critical)

Discard any 200bp sequence containing **at least one `N` character**. The competition dataset had zero N's; your training data must match this constraint for the model to generalize to competition test sequences.

N characters appear in the hg38 reference at:
- Centromeric regions
- Telomeric regions
- Assembly gaps (regions sequenced insufficiently in hg38)
- Short gaps within otherwise assembled regions

**Expected N-rate:** Roughly 5–8% of bins after liftover, concentrated in specific chromosomal regions. Loss is not uniformly distributed — chrY and pericentromeric regions of large chromosomes will have disproportionately high N-rates.

### Low-Complexity Filtering (Optional but Recommended)

Sequences with extremely low Shannon entropy are nearly all one nucleotide (e.g., poly-A or poly-T runs). These occur primarily in heterochromatin states and can cause numerical issues during training (near-zero gradient on trivially classified sequences).

**Shannon entropy per sequence:** Compute the base frequency vector `[p_A, p_C, p_G, p_T]` and calculate `H = -Σ p_i * log2(p_i)`. For a 200bp sequence, a reasonable minimum threshold is `H > 0.5 bits`. Sequences below this are extreme low-complexity and can be dropped. Log the drop rate per state — if state 13 (Het) drops 30% of sequences here, that's biologically expected (heterochromatin is repeat-rich).

### Sequence Length Validation

After extraction, assert all sequences are exactly 200 characters long. Any deviation indicates a FASTA extraction error, typically at chromosome boundaries. Drop and log.

### Valid Alphabet Check

Sequences should contain only `{A, C, G, T}` after N-filtering. Check for any other characters (sometimes `R`, `Y`, `S`, `W` IUPAC ambiguity codes appear in some assemblies). These are rare in hg38 but worth filtering explicitly.

---

## 8. State Label Normalization

### String-to-Integer Mapping

The Roadmap BED files use string state names (`1_TssA`, `7_Enh`, etc.). Convert to integers by stripping the underscore-separated name and keeping the numeric prefix. The canonical mapping:

| Integer | State Name | Biological Role |
|---------|-----------|----------------|
| 1 | TssA | Active TSS |
| 2 | TssFlnk | Flanking TSS |
| 3 | TssFlnkU | Flanking TSS upstream |
| 4 | TssFlnkD | Flanking TSS downstream |
| 5 | Tx | Strong transcription |
| 6 | TxWk | Weak transcription |
| 7 | EnhG1 | Genic enhancer 1 |
| 8 | EnhG2 | Genic enhancer 2 |
| 9 | EnhA1 | Active enhancer 1 |
| 10 | EnhA2 | Active enhancer 2 |
| 11 | EnhWk | Weak enhancer |
| 12 | ZNF/Rpts | ZNF genes & repeats |
| 13 | Het | Heterochromatin |
| 14 | TssBiv | Bivalent/poised TSS |
| 15 | EnhBiv | Bivalent enhancer |
| 16 | ReprPC | Repressed Polycomb |
| 17 | ReprPCWk | Weak repressed Polycomb |
| 18 | Quies | Quiescent/low |

**Label convention:** Use 1-indexed integers to match the competition format. Store as integers, not strings, in all output files.

### Biological Hierarchy Encoding

Since you already derived this in Phase 8, encode the family hierarchy directly into the dataset metadata. Add two additional columns alongside the integer label:

| Column | Values | Description |
|--------|--------|-------------|
| `family` | `{promoter, transcribed, enhancer, polycomb, heterochromatin, quiescent}` | Coarse biological group |
| `subcluster` | Integer 1–7 | Fine subgroup within family |

These columns are not used by the baseline model but are immediately available for hierarchical loss functions, family-level accuracy tracking, and steering experiments — all of which your writeup identified as important.

---

## 9. Cross-Epigenome Deduplication

### The Core Problem

Every genomic locus in the whitelist chromosomes will appear in all epigenomes you download, lifted to the same hg38 coordinates. If you naively concatenate 30 epigenomes, each unique locus appears 30 times. Whether this is a bug or a feature depends on your research goals.

### Option A: Retain All Copies (Recommended for MI Research)

Keep all (sequence, label) pairs regardless of coordinate duplication. A sequence from `chr1:1000-1200` labeled state 1 (TssA) in E003 and state 18 (Quies) in E034 are two genuinely distinct training examples — they reflect that the same DNA sequence is in a different chromatin state in different cell types. This is biologically real variance.

**Implications:**
- Model sees the same sequence with potentially contradictory labels → learns sequence features that are robustly predictive across cell types, not cell-type-specific features
- For MI research, this is desirable: you want the model to learn what DNA sequence features are *intrinsically* associated with each state
- Add an `EID` column to track which epigenome each row came from — this enables post-hoc cell-type analysis

### Option B: Coordinate-Level Deduplication

For each unique hg38 coordinate, assign the plurality label (most frequent state across all epigenomes at that locus). Ties broken by biological priority order (active states > repressed > quiescent).

**Implications:**
- Cleaner label signal for sequences where most epigenomes agree
- Loses cell-type-specific examples where a locus is TssA in one tissue and Quies in most others
- Reduces dataset size proportionally to number of epigenomes merged

### Option C: Locus-Level Variance Filtering

Keep only loci where all sampled epigenomes agree on the state. This gives the highest-confidence labels but aggressively reduces dataset size and biases heavily toward quiescent and heterochromatic regions (which are state-stable across cell types).

**Not recommended** for this project — the biologically interesting states (TssA, enhancers) are precisely those that vary across cell types.

---

## 10. Dataset Balancing

### The Imbalance Problem

The raw Roadmap data is heavily skewed toward quiescent states. Typical genome-wide distribution:

| State Group | Approximate Coverage |
|-------------|---------------------|
| Quies (18) | 50–70% |
| Het (13) | 10–20% |
| Tx, TxWk (5,6) | 5–10% |
| Active promoters, enhancers | 1–5% |
| All others | <1% each |

Without balancing, a trivial model that predicts Quies for everything achieves >50% accuracy. This is a degenerate solution and occurred implicitly in the hackathon model's behavior.

### Balancing Strategies

**Strategy 1 — Hard cap per state (match competition distribution):**  
Set a maximum N samples per state. Competition used 15,898 per state. For the full Roadmap corpus, N = 50,000 per state is a reasonable starting point that gives the model enough variation while being computationally tractable. Sample randomly within each state across all epigenomes.

**Strategy 2 — Stratified cap per state per epigenome:**  
Cap each (epigenome × state) combination independently, then pool. This ensures biological diversity — you don't accidentally sample all your TssA examples from a single epigenome. Cap formula: `N_global / N_epigenomes` per cell, with remainder filled from whichever epigenomes have the most examples.

**Strategy 3 — Class-weighted loss (no resampling):**  
Preserve the raw distribution in the dataset but compute per-class weights inversely proportional to frequency. Weight for state k: `w_k = N_total / (18 × N_k)`. Pass these weights to the loss function during training. This is the most principled approach statistically but requires implementation in the training loop.

**Recommendation:** Use Strategy 2 for the dataset files, and additionally implement Strategy 3 in the training loop. The two are complementary — resampling prevents the data loader from being dominated by quiescent sequences, while class weighting corrects any residual imbalance in the sampled distribution.

### Minority State Augmentation

States 14 (TssBiv), 15 (EnhBiv), and 12 (ZNF/Rpts) are rare genome-wide. After balancing, you may not have enough examples even at N=50,000. For these states:

- Pool across all 30 epigenomes before capping (do not cap per epigenome)
- If still underrepresented, apply reverse-complement augmentation: for every sequence from a minority state, also include its reverse complement with the same label. This doubles minority state samples while adding biologically valid training examples (since ChromHMM states are strand-symmetric)

---

## 11. Train / Val / Test Split

### Split by Chromosome (Not Random)

**Do not randomly shuffle and split.** Genomic sequences from nearby loci share local sequence context (repetitive elements, GC-content domains, regional chromatin domains). A random split will allow the model to memorize local sequence neighborhood patterns, producing inflated validation accuracy that doesn't generalize. This is a well-known issue in genomic deep learning.

**Standard chromosome holdout:**

| Split | Chromosomes | Approximate Coverage |
|-------|------------|---------------------|
| Training | chr1–chr17 | ~75% of genome |
| Validation | chr18, chr19 | ~8% of genome |
| Test | chr20, chr21, chr22 | ~10% of genome |

This follows the convention established by DeepSEA (Zhou & Troyanskaya, 2015) and widely adopted since.

**Competition test set:** Retain the original `testsequences.csv` from the competition as an additional held-out benchmark. This lets you directly compare your new model's performance against the hackathon baseline on the same test sequences.

### Chromosome Split Caveats

- chrX and chrY: include in training only. Sex chromosome biology is unusual and their validation behavior may be misleading.
- Check that all 18 states are represented in both validation and test sets after the split. Rare states concentrated in specific chromosomal regions might have zero representation in chr20-22 — if so, adjust the split to include one chromosome where they're more common.

---

## 12. Quality Checks & Sanity Metrics

Run these checks after each major pipeline stage and log results to `logs/qc/`.

### Per-Epigenome QC Table

After full processing, for each epigenome generate a summary row:

| Metric | Expected Range | Action if Outside |
|--------|---------------|------------------|
| Raw BED rows | 14M–16M | Investigate file |
| Rows surviving normalization | >99% | Check chromosome list |
| Rows surviving liftover | >85% | Check chain file |
| Rows surviving N-filter | >90% | Check hg38 gaps |
| Rows surviving length check | >99.9% | Check extraction tool |
| State vocabulary complete | All 18 states present | Flag epigenome |
| Quies fraction | 50–70% | Verify state mapping |
| TssA fraction | 0.5–3% | Verify state mapping |

### Cross-Epigenome Consistency Check

For any two epigenomes from the same tissue group (e.g., two blood epigenomes), the per-state sequence-level GC content distributions should be nearly identical. A large divergence (>0.05 mean GC difference for TssA between two blood epigenomes) suggests a labeling or mapping error.

### Label Distribution Verification

After balancing, plot the final per-state counts and verify:
- All 18 states present
- No state has zero examples in validation or test split
- Quies does not dominate (should be at or near target N after balancing)

### Reverse-Complement Consistency Spot-Check

For a random sample of 1,000 sequences, verify that the sequence extracted for `chr1:1000-1200` and the reverse complement of the sequence extracted for `chr1:1000-1200` are consistent with the reference genome. This catches FASTA extraction errors.

### Competition Format Compatibility Check

Extract 100 random rows from your final `sequences.csv` and verify:
- Exactly 200 characters per row
- Only `{A, C, G, T}` characters
- No header row
- Integer labels in `labels.csv` are in range [1, 18]

These are the exact constraints the competition scorer expects. Failures here will cause silent scoring errors.

---

## 13. Final Schema

### Per-Epigenome Files

Two files per epigenome, N rows each, aligned by line number:

**`E003_sequences.csv`** — one 200bp sequence per line, no header:
```
ACGTACGT...  (200 chars)
GCTTAGCA...  (200 chars)
...
```

**`E003_labels.csv`** — one integer per line, no header:
```
1
18
7
...
```

### Merged Training Corpus

After pooling across epigenomes and balancing:

**`train_sequences.csv`** — all training sequences (chromosomes 1-17), balanced  
**`train_labels.csv`** — corresponding integer labels  
**`train_meta.csv`** — one row per training sequence with columns:

| Column | Type | Description |
|--------|------|-------------|
| `eid` | string | Source epigenome ID (e.g., E003) |
| `chrom` | string | Source chromosome in hg38 |
| `start` | int | 0-based start coordinate in hg38 |
| `end` | int | End coordinate (always start+200) |
| `state_int` | int | Integer label [1-18] |
| `state_name` | string | Human-readable state name |
| `family` | string | Biological family group |
| `subcluster` | int | Fine subcluster index |
| `gc_content` | float | GC fraction of this sequence |
| `cpg_ratio` | float | Observed/expected CpG ratio |
| `entropy` | float | Shannon entropy in bits |

The metadata file is not fed to the model — it's retained for MI analysis, debugging, and stratified sampling experiments.

Analogous files for `val_` and `test_` splits.

---

## 14. Epigenome Selection Reference

Priority epigenomes for biological diversity, covering major tissue groups from the Roadmap project. This 30-epigenome subset is recommended for an initial full pipeline run.

| EID | Cell/Tissue Type | GROUP | Priority |
|-----|-----------------|-------|----------|
| E003 | H1 Embryonic Stem Cell | ESC | High |
| E008 | H9 Embryonic Stem Cell | ESC | Medium |
| E020 | iPS-20b | iPSC | High |
| E023 | Mesenchymal Stem Cell | ES-deriv | Medium |
| E032 | Primary B cells from blood | Blood | High |
| E034 | Primary T cells from blood | Blood | High |
| E040 | Primary Monocytes from blood | Blood | Medium |
| E055 | Foreskin Fibroblast Primary | Mesench | High |
| E065 | Aorta | Heart | Medium |
| E066 | Liver | Digestive | High |
| E069 | Brain Prefrontal Cortex | Brain | High |
| E071 | Brain Hippocampus Middle | Brain | Medium |
| E080 | Fetal Adrenal Gland | Other | Low |
| E086 | Fetal Kidney | Other | Medium |
| E094 | Gastric | Digestive | Low |
| E095 | Left Ventricle | Heart | High |
| E096 | Lung | Other | High |
| E100 | Psoas Muscle | Muscle | High |
| E108 | Skeletal Muscle Male | Muscle | High |
| E114 | A549 Lung Carcinoma | ENCODE | Low |
| E116 | GM12878 Lymphoblastoid | ENCODE | High |
| E117 | IMR90 Fetal Lung Fibroblast | ENCODE | Medium |
| E118 | HepG2 Liver Carcinoma | ENCODE | High |
| E119 | HMEC Mammary Epithelial | Epithelial | High |
| E120 | HSMM Skeletal Myoblast | Muscle | High |
| E122 | HUVEC Umbilical Vein | Other | High |
| E123 | K562 Leukemia | ENCODE | High |
| E126 | NHEK Keratinocyte | Epithelial | High |
| E127 | NHDF-Ad Fibroblast | Mesench | Medium |
| E128 | NHLF Lung Fibroblast | Other | Medium |

**ENCODE-tier epigenomes** (E114–E128) share cell lines with the original ChromHMM paper and the hackathon challenge — include these for direct comparability with your baseline results.

---

*Guide version: April 2026 | Genome assembly: hg38 | ChromHMM model: Roadmap 18-state core marks*
