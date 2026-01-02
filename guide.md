# Chromatin State Prediction Challenge: Research-Oriented Master Plan

## Philosophy

This plan prioritizes **mechanistic understanding** and **interpretability** over raw performance optimization. The goal is not just to predict labels accurately, but to discover what biological patterns the model learns and to develop alignment techniques that could generalize beyond this competition.

Notes: 

1. All computational intensive code should utilize the mac m1 multi-core cpu. Alternatively MPS for parallel computing.
2. Use demo datasets in "data" folder for all testings and initial runnings to quickly debug.

---

## Phase 1: Data Engineering & Biological Priors

### 1.1 Sequence Encoding Strategies

**Primary Encoding: One-Hot**

- Shape: (200, 4) where channels represent A, C, G, T
- Preserves positional information critical for motif detection
- Compatible with convolutional operations

**Secondary Encoding: Learnable Embeddings**

- Map each nucleotide to a learnable d-dimensional vector (d=8 or 16)
- Allows the model to discover latent relationships between bases
- Useful for interpretability: examine what embedding space reveals about base similarities

**Tertiary Encoding: K-mer Tokenization(optional)**

- Treat sequence as series of overlapping k-mers (k=3 or 4)
- Vocabulary size: 64 (3-mers) or 256 (4-mers)
- Each token gets an embedding; enables transformer-style architectures if desired later

### 1.2 Augmentation Protocol

**Reverse Complement (RC) Augmentation**

- Biological rationale: DNA is double-stranded; a motif on the forward strand appears as its RC on the reverse strand
- Implementation: For sequence S, compute RC(S) by complementing (A↔T, C↔G) and reversing
- Training protocol: Each batch contains both S and RC(S) with the same label
- Critical constraint for interpretability: The model should learn **RC-equivariant** representations — activations for S and RC(S) should be related by a predictable transformation

**Position Jittering**

- Randomly crop a 180-190bp window from the 200bp sequence
- Pad back to 200bp with random flanking sequence or zeros
- Forces the model to recognize motifs regardless of absolute position
- Controlled via hyperparameter: jitter probability (suggest 0.3-0.5 during training)

**Noise Injection**

- With low probability (1%), substitute a random nucleotide
- Simulates sequencing errors and biological variation
- Should improve generalization without harming interpretability

### 1.3 Stratified Cross-Validation Framework

- 5-fold stratified split preserving 1/18 class balance per fold
- Designate fold 5 as a held-out "interpretability test set" — never used for training or hyperparameter selection
- Use folds 1-4 for training/validation cycles. 1-3 for training, 4 for validation.
- All interpretability experiments run on fold 5 to ensure findings generalize

---

## Phase 2: Manifold Learning & Visualization

### 2.1 Feature Extraction for Visualization

**K-mer Frequency Vectors**

- Compute frequency of all 5-mers (1,024 features) or 6-mers (4,096 features) for each sequence
- Normalize to relative frequencies (sum to 1)
- This creates a "bag of words" representation of local sequence composition

**Positional K-mer Profiles**

- Divide 200bp into 10 bins of 20bp each
- Compute k-mer frequencies within each bin
- Concatenate to get position-aware features (10 × 1,024 = 10,240 features for 5-mers)
- Captures positional biases (e.g., TATA boxes occur ~25-30bp upstream of transcription start)

**Dinucleotide Transition Frequencies**

- Count all 16 possible dinucleotide transitions (AA, AC, AG, AT, CA, ...)
- Captures local correlations and stacking energies

### 2.2 Dimensionality Reduction Methods

**UMAP (Uniform Manifold Approximation and Projection)**

- Parameters to explore: n_neighbors (5, 10, 15, 20, 30, 50), min_dist (0.0, 0.1, 0.25, 0.5)
- Run on k-mer frequency vectors
- Produces 2D or 3D embeddings for visualization

**PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)**

- Better at preserving global structure and continuous trajectories
- Particularly useful if chromatin states form a biological continuum rather than discrete clusters
- Parameters: knn (5, 10, 15), decay (10, 20, 40)

**PCA (as baseline)**

- Examine variance explained by top components
- First 2-3 PCs may capture gross compositional differences (GC content, repeat content)

### 2.3 Visualization Protocol

**Static Visualizations: simple dashboard**

- 2D scatter plots colored by label (1-18), using consistent color palette
- Compute and display cluster centroids for each label
- Draw convex hulls or density contours around each label's points

**Quantitative Cluster Analysis**

- Compute silhouette scores for the 18-class labeling in each embedding space
- Compute adjusted Rand index between k-means clustering (k=18) and true labels
- Identify which labels have high intra-class cohesion vs. which are dispersed

**Pairwise Label Relationships**

- Compute centroid-to-centroid distances between all pairs of labels
- Build a 18×18 distance matrix
- Perform hierarchical clustering on this matrix to discover label super-groups
- Visualize as a dendrogram

**Confusion Prediction**

- Labels with nearby centroids in manifold space are likely to be confused by the model
- Use this to predict which class pairs will have high confusion rates
- Later validate against actual model confusion matrix

### 2.4 Biological Interpretation of Manifold Structure

**GC Content Gradient**

- Color points by GC content (proportion of G+C nucleotides)
- Determine if manifold axes correlate with compositional biases

**Repeat Content**

- Scan for simple repeats (poly-A, poly-T, dinucleotide repeats)
- Color points by repeat density
- Some chromatin states (heterochromatin) are repeat-rich

**Known Motif Scores**

- Score each sequence for presence of known regulatory motifs (TATA, CAAT, GC-box, E-box)
- Color manifold by motif scores
- Determine if certain labels cluster around motif-rich regions

---

## Phase 3: Interpretability-Friendly CNN Architecture

Pytorch

### 3.1 Design Principles for Mechanistic Interpretability

**Principle 1: Minimize Polysemanticity**

- Polysemantic neurons respond to multiple unrelated features
- Reduce by: using wider layers (more neurons), encouraging sparsity, avoiding deep bottlenecks

**Principle 2: Preserve Spatial Information**

- Avoid aggressive pooling early in the network
- Use global pooling only at the final convolutional layer
- This allows tracing which sequence positions activate which features

**Principle 3: Modular Structure**

- Clear separation between "motif detection" (conv layers) and "decision making" (dense layers)
- Easier to interpret: conv filters = motifs, dense layers = combinatorial logic

**Principle 4: RC Equivariance**

- Architecture should treat S and RC(S) symmetrically
- Option A: RC augmentation during training (soft constraint)
- Option B: Explicitly RC-equivariant architecture (hard constraint) — average activations from both strands

### 3.2 Recommended Architecture

```
Input: (batch, 200, 4) one-hot encoded

═══════════════════════════════════════════════════
MOTIF DETECTION BLOCK (Interpretable)
═══════════════════════════════════════════════════
[Conv1D] 128 filters, kernel_size=19, padding='same', ReLU
         → Captures motifs up to 19bp (typical TF binding site length)
         → Output: (batch, 200, 128)
       
[BatchNorm]
         → Stabilizes training; can be folded into conv for interpretation

[Conv1D] 256 filters, kernel_size=11, padding='same', ReLU  
         → Captures motif combinations and longer patterns
         → Output: (batch, 200, 256)

[BatchNorm]

═══════════════════════════════════════════════════
SPARSE BOTTLENECK (SAE Attachment Point)
═══════════════════════════════════════════════════
[Conv1D] 512 filters, kernel_size=1, ReLU
         → 1x1 convolution acts as position-wise feature mixing
         → This is where SAE will be attached
         → Output: (batch, 200, 512)

═══════════════════════════════════════════════════
SPATIAL AGGREGATION
═══════════════════════════════════════════════════
[Global Max Pooling]
         → Extracts strongest activation per filter across all positions
         → Output: (batch, 512)
         → Key for interpretability: we can recover WHICH position fired maximally

[Global Average Pooling] (parallel branch)
         → Captures overall presence of a motif
         → Output: (batch, 512)

[Concatenate] → (batch, 1024)

═══════════════════════════════════════════════════
DECISION BLOCK
═══════════════════════════════════════════════════
[Dense] 512 units, ReLU, Dropout(0.3)
[Dense] 256 units, ReLU, Dropout(0.3)
[Dense] 18 units, Softmax → Output class probabilities
```

### ~~3.3 Architectural Variants for Comparison~~

**Variant A: Shallow-Wide**

- Single conv layer with 512 filters of size 19
- Direct global pooling to dense layers
- Maximum interpretability: each filter = one motif concept

**Variant B: Deep-Narrow**

- 5-6 conv layers with 64-128 filters each
- Residual connections
- Higher capacity but harder to interpret

**Variant C: Attention-Augmented**

- After conv layers, add self-attention over positions
- Attention weights reveal which positions interact
- Useful for discovering long-range dependencies

**Recommendation:** Start with the main architecture, establish baseline interpretability findings, then compare with variants.

### 3.4 Training Protocol

**Optimizer:** AdamW with weight decay 1e-4

**Learning Rate Schedule:**

- Warmup: Linear increase from 1e-5 to 1e-3 over first 5 epochs
- Main: Cosine annealing from 1e-3 to 1e-6 over remaining epochs

**Batch Size:** 256 (adjust based on GPU memory)

**Epochs:** 50-100 with early stopping (patience=10 on validation loss)

**Loss Function:**

- Primary: Cross-entropy
- Consider: Label smoothing (ε=0.05) to prevent overconfident predictions, which aids interpretability

**Regularization:**

- Dropout as specified
- L1 penalty on first conv layer filters (encourages sparse, interpretable motifs)
- Gradient clipping (max norm=1.0)

---

## Phase 4: Mechanistic Interpretability — Foundation

PyTorch hooks + Nnsight

### 4.1 Filter Visualization & Motif Extraction

**Objective:** Convert learned conv filters into human-interpretable motifs.

**Procedure for First Conv Layer:**

1. Extract the 128 filters of shape (19, 4) from the first conv layer
2. For each filter, compute its Position Weight Matrix (PWM):
   - Normalize filter weights to sum to 1 at each position
   - Treat as log-odds scores; convert to probabilities via softmax
3. Visualize as sequence logos:
   - Use standard logo visualization (letter height = information content)
   - Generate logo for each of the 128 filters
4. Compute Information Content (IC) per filter:
   - IC = sum over positions of KL divergence from uniform
   - High IC filters have strong sequence preferences (likely functional)
   - Low IC filters are "noise" or compositional detectors
5. Cluster similar filters:
   - Compute pairwise distances between filter PWMs
   - Hierarchical clustering to group redundant motifs
   - May find that 128 filters collapse to 30-50 distinct motif families

**Procedure for Deeper Conv Layers:**

- Direct visualization is less meaningful (filters operate on feature space, not sequence space)
- Instead, use activation maximization (see 4.2)

### 4.2 Activation Maximization

**Objective:** Find synthetic sequences that maximally activate each filter or neuron.

**Procedure:**

1. Initialize a random one-hot sequence (200, 4)
2. Forward pass through network to target layer/neuron
3. Compute gradient of target activation with respect to input
4. Update input via gradient ascent:
   - Add gradient × step_size to input
   - Project back to valid one-hot via softmax at each position
5. Iterate for 100-500 steps
6. The resulting sequence is "what this neuron is looking for"

**Variations:**

- **Regularized activation maximization:** Add penalty for low-complexity sequences (avoids poly-A solutions)
- **Diverse activation maximization:** Generate multiple solutions by penalizing similarity to previous solutions
- **Natural priors:** Initialize from real training sequences rather than random

**Analysis:**

- Run activation maximization for all 128 first-layer filters
- Run for all 256 second-layer filters
- Run for the 512 bottleneck features
- Cluster resulting sequences to identify distinct motif families

### 4.3 Dataset Examples Analysis

**Objective:** Understand filters via the real sequences that activate them.

**Max Activating Examples:**

1. For each filter in the bottleneck layer (512 filters):
   - Pass all training sequences through the network
   - Record the activation value for this filter (max across positions)
   - Collect the top 100 sequences with highest activation
2. For each filter, analyze its top-100 sequences:
   - What labels are represented? (filter → label association)
   - What k-mers are enriched compared to background?
   - What positions within the 200bp show highest activation?
3. Build a filter-to-label association matrix (512 × 18):
   - Entry (i, j) = proportion of filter i's top examples belonging to label j
   - Filters with peaked distributions are "label-specific"
   - Filters with flat distributions are "general" (maybe GC content, etc.)

**Min Activating Examples (Counterfactuals):**

- Also collect sequences with near-zero activation
- Compare max vs. min: what is present in max that is absent in min?

### 4.4 Attribution via DeepLIFT

**Objective:** For a given prediction, identify which input positions were important.

**DeepLIFT Overview:**

- DeepLIFT (Deep Learning Important FeaTures) computes contribution scores by comparing activations to a reference input
- Unlike gradient-based methods, DeepLIFT assigns discrete credit for differences from baseline
- Handles saturation in ReLU networks better than vanilla gradients

**Reference Input Selection:**

- Option A: All-zeros input (simple but may not be biologically meaningful)
- Option B: Uniform 0.25 across all positions and nucleotides (represents "no information")
- Option C: Shuffled version of the actual sequence (preserves composition)
- Recommendation: Use Option B as primary, validate with Option C

**DeepLIFT Rules:**

- **Linear Rule:** For linear layers, contribution is proportional to weight × (activation - reference_activation)
- **Rescale Rule:** For nonlinear layers (ReLU), rescale contributions based on activation differences
- **RevealCancel Rule:** Separates positive and negative contributions (more precise but computationally heavier)
- Recommendation: Start with Rescale Rule; use RevealCancel for detailed analysis of specific examples

**Execution Procedure:**

1. Select sequences for analysis:
   - 500 random sequences per class (9,000 total for all 18 classes)
   - Include both high-confidence and borderline predictions
2. For each sequence:
   - Compute DeepLIFT attribution scores with respect to the predicted class logit
   - Output shape: (200, 4) — contribution of each nucleotide at each position
   - Sum across nucleotides to get positional importance: (200,)
3. Aggregate attribution patterns per class:
   - Average positional importance across all sequences of each class
   - Identify class-specific "hot spots" where position consistently matters
4. Visualize attributions:
   - Plot attribution heatmaps for individual sequences
   - Overlay on sequence to show which nucleotides drove the prediction
   - Create aggregate positional profiles per class

**Validation of Attributions:**

1. **Sanity checks:**
   - Attributions should sum approximately to (prediction - baseline_prediction)
   - High-attributed positions should correspond to known motif locations
2. **Perturbation consistency:**
   - Take top-10 attributed positions for a sequence
   - Mutate these positions randomly
   - Verify that prediction changes significantly (attributions are faithful)
3. **Comparison across methods:**
   - Optionally run vanilla gradients or gradient × input
   - Check correlation with DeepLIFT attributions
   - DeepLIFT should be more stable and less noisy

**Analysis Outputs:**

1. **Per-class positional profiles:**
   - 18 plots showing average attribution by position
   - Do certain classes show strong positional preferences?
2. **Motif attribution:**
   - For sequences containing known motifs, do attributions highlight the motif region?
   - Quantify: what fraction of total attribution falls within motif boundaries?
3. **Attribution clustering:**
   - Cluster sequences by their attribution patterns (not just their nucleotide content)
   - Do clusters correspond to labels? (They should, if attributions are meaningful)

### 4.5 Probing Classifiers

**Objective:** Test what information is linearly accessible at each layer.

**Procedure:**

1. Freeze the trained CNN
2. Extract activations at each layer for all training sequences:
   - After Conv1: (200, 128) → flatten or pool to fixed size
   - After Conv2: (200, 256)
   - After bottleneck: (200, 512)
   - After global pool: (1024,)
3. Train simple linear probes (logistic regression) to predict:
   - The 18-class label
   - GC content (regression)
   - Presence/absence of specific k-mers
   - Position of strongest motif match
4. Measure probe accuracy at each layer

**Interpretation:**

- If label prediction probe accuracy increases through layers → network progressively builds label-relevant features
- If GC content probe is highly accurate at layer 1 but not used for labels → network learns but discards compositional information
- Probes reveal the "representational geometry" at each layer

---

## Phase 5: Sparse Autoencoders for Feature Decomposition

### 5.1 Motivation

**The Superposition Hypothesis:**

- Neural networks represent more features than they have neurons
- Features are encoded in "superposition" — overlapping patterns across neurons
- This creates polysemanticity: individual neurons respond to multiple unrelated concepts

**SAE Solution:**

- Train an autoencoder with a much wider hidden layer (e.g., 512 → 4096)
- Enforce sparsity: only a few hidden units active for any input
- Result: hidden units become "monosemantic" — each represents one clean feature

### 5.2 SAE Architecture

```
Input: Bottleneck activations (batch, 200, 512) or pooled (batch, 512)

[Encoder]
Linear: 512 → 4096 (expansion factor 8x)
ReLU activation (enforces non-negativity)

[Sparsity Bottleneck]
Top-K sparsity: keep only K highest activations, zero others
OR L1 penalty on activations

[Decoder]  
Linear: 4096 → 512 (reconstruct original)
No activation (linear reconstruction)
```

### 5.3 SAE Training Protocol

**Loss Function:**

```
L = L_reconstruction + λ × L_sparsity

L_reconstruction = MSE(input, reconstruction)
L_sparsity = mean(|hidden_activations|)  [L1 penalty]
```

**Hyperparameters to Tune:**

- Expansion factor: 4x, 8x, 16x (wider = more features, but harder to train)
- Sparsity coefficient λ: 1e-3 to 1e-1 (higher = sparser)
- Top-K value: 10, 20, 50 (if using top-K instead of L1)

**Training Procedure:**

1. Freeze the base CNN
2. Pass all training data through CNN, collect bottleneck activations
3. Train SAE on these activations (separate from CNN training)
4. Iterate: adjust λ until achieving target sparsity (e.g., 95% zeros)

**Evaluation Metrics:**

- Reconstruction loss: How well does SAE preserve information?
- Sparsity: What fraction of hidden units are zero on average?
- Dead neurons: What fraction of hidden units never activate? (should be low)

### 5.4 SAE Feature Analysis

**Feature-to-Label Mapping:**

1. For each of the 4096 SAE features:
   - Collect all training sequences where this feature activates (above threshold)
   - Compute label distribution among these sequences
   - Assign "primary label" = most common label
2. Build feature-label association matrix (4096 × 18)
3. Identify:
   - Label-specific features (activate almost exclusively for one label)
   - Shared features (activate across multiple related labels)
   - Universal features (activate for all labels — likely compositional)

**Feature Interpretability:**

1. For each SAE feature, collect top-50 max-activating sequences
2. Extract the 19bp window centered on the max-activation position
3. Run motif discovery on these windows:
   - Compute position weight matrix from alignment
   - Compare to known motif databases
   - Assign human-readable interpretation (e.g., "Feature 2847 = CTCF binding site")
4. Create a "feature dictionary" mapping feature index → biological interpretation

**Feature Co-occurrence Analysis:**

1. For each training sequence, record which SAE features are active
2. Compute co-occurrence matrix: how often do features i and j activate together?
3. Cluster features by co-occurrence patterns
4. Interpretation: co-occurring features may represent parts of the same biological process

### 5.5 SAE Validation

**Reconstruction Quality:**

- After SAE compression, does the CNN still predict correctly?
- Replace bottleneck activations with SAE reconstructions
- Measure accuracy drop (should be minimal if SAE preserves key information)

**Feature Necessity:**

- Ablate individual SAE features (set to zero)
- Measure prediction change
- Features that cause large changes when ablated are "necessary"

**Feature Sufficiency:**

- Activate only a single SAE feature (zero all others)
- Does this produce a confident prediction for a specific label?
- Features that are sufficient alone are "atomic concepts"

---

## Phase 6: Steering & Alignment Techniques

### 6.1 Representation Engineering

**Objective:** Control model behavior by intervening on internal representations.

**Computing Steering Vectors:**

1. Collect all training sequences for Label X
2. Pass through network, extract activations at target layer (e.g., bottleneck)
3. Compute mean activation: μ_X = mean(activations for Label X)
4. Repeat for Label Y: μ_Y = mean(activations for Label Y)
5. **Steering vector:** v_{X→Y} = μ_Y - μ_X

**Applying Steering:**

1. Take a new sequence predicted as Label X
2. At target layer, add steering vector: activation_new = activation_old + α × v_{X→Y}
3. Continue forward pass
4. Observe: does prediction shift toward Y?

**Calibrating Steering Strength (α):**

- α = 0: no effect
- α = 1: full effect (may overshoot)
- Find optimal α via validation set

### 6.2 Activation Addition for Robustness

**Objective:** Make predictions more robust by amplifying correct features.

**Procedure:**

1. Identify SAE features that are diagnostic for each label (from Phase 5)
2. At inference time:
   - Run sequence through network
   - If prediction is uncertain (low confidence), identify which diagnostic features are weakly active
   - Add activation to strengthen these features
   - Re-run forward pass
3. This "boosts" the signal for the features the model should be using

**Validation:**

- Test on held-out fold
- Measure: does activation addition improve accuracy on uncertain predictions?
- Ensure it doesn't hurt accuracy on confident predictions

### 6.3 Contrastive Activation Addition

**Objective:** Suppress confusions between specific label pairs.

**Setup:** Labels A and B are frequently confused (from confusion matrix)

**Procedure:**

1. Compute steering vector v_{A→B} and v_{B→A}
2. For sequences where model is uncertain between A and B:
   - Examine which steering vector, when added, increases confidence
   - The correct label is the one where adding v_{X→X} (identity boost) increases confidence
3. Implement as inference-time correction

### 6.4 Alignment Evaluation

**Test Suite:**

1. **Consistency:** Does the model give the same prediction for S and RC(S)?
   - Measure RC consistency rate
   - Use steering to improve if needed
2. **Calibration:** Do model confidences match actual accuracies?
   - Plot reliability diagram
   - Use temperature scaling to calibrate
3. **Monotonicity:** If we strengthen a motif, does confidence in the associated label increase monotonically?
   - Create synthetic sequences with varying motif strength
   - Test that predictions behave sensibly
4. **Compositionality:** Does the model handle combinations of motifs correctly?
   - Create sequences with multiple motifs
   - Verify predictions reflect motif combination

---

## Deliverables & Documentation

### For Competition Submission

- `predictions.csv` generated from best-performing model configuration
- Ensure test-time augmentation (RC averaging) is applied

### For Research Documentation

**Model Card:**

- Architecture details
- Training procedure
- Validation accuracy per class
- Known failure modes

**Feature Dictionary:**

- All 128 first-layer filters with sequence logos
- All 4096 SAE features with interpretations (where discovered)
- Feature-to-label association mappings

**Alignment Report:**

- RC consistency metrics
- Calibration curves
- Steering vector effectiveness

**Manifold Analysis Report:**

- UMAP/PHATE visualizations with label coloring
- Label cluster analysis and dendrogram
- Predicted vs. actual confusion patterns
