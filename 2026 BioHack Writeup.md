# 2026 BioHack Writeup: Mechanistic Interpretability On Bio application models

**Proposed Pipeline**

Phase 1: Genomic Data Engineering & Augmentation
Phase 2: Exploratory Manifold Analysis, dimensional reduction
Phase 3: Model: CNN
~~Phase 4: Causal Patching~~ (didn't actually implement for decreased return ratio)
~~Phase 5: Sparse Autoencoders for Feature Decomposition~~ (didn't actually implement for decreased return ratio)
Phase 6: Steering for Performance (Inference-Time Intervention)
Phase 7: Diagnostics & Saliency
Phase 8: Hacking (Enriching Metadata, then feed the dataset and classification hiearchy back into Phase 3, 6 & 7)



**Focus:**

Have fun and be comfortable with bold experimentations: Data insights + Mechanistic Interpretability -> Comp Bio

Intuitions on dataset: seeing the landscape of the data through manifold learning and dimension reductions

Mechanistic Interpretability: Understand causation. How, What and Why the model works

Hacking: Augementation of the dataset and bringing more Genetics context to the dataset



**Methodology:** 

Me as the head engineer/scientist know every detail in first principle. Drafting and refining plan. Researching. Reading, debugging code

Claude sonnet: actually write most of the code



**Highlights:**

### Phase 2: Exploratory Manifold Analysis
- Goal: understand dataset geometry before modeling; grouping labels on a manifold
- Representation: 6-mer frequency vectors (4096-d) as "local vocabulary"
- Methods: UMAP + PHATE (2D projections), plus PCA as baseline. Note: dimensions in UMAP has no meaningful unit
- no 18 clean clusters; continuous landscape with gradients
- Key axis: GC-content gradient (promoter-like vs heterochromatin-like), yet still minority contribution

![pca_2d](file:///Users/xiaoyuwang/Desktop/manifold-dim-reduct/results/phase2_manifold/figures/pca_2d.png)

![umap_n15_d0.1_2d](file:///Users/xiaoyuwang/Desktop/manifold-dim-reduct/results/phase2_manifold/figures/umap_n15_d0.1_2d.png)



### Phase 3: CNN
- Goal: transparency-first model; easy to interpret motifs and decisions
- Core: 1D-CNN motif scanners (first conv: 128 filters, kernel ~19bp)
- Pooling: global max pooling for position invariance (paired with jittering/RC augmentation)
- Bottleneck layer: explicit 256-d bottleneck after global pooling, chosen as the activation extraction point for all downstream MI analysis (steering vectors, linear probes, saliency)



### Phase 4 & 5: ~~Sparse Autoencoder (SAE) Feature Decomposition~~ & ~~Causal Patching~~ (Not Implemented)

**What SAEs are:** A Sparse Autoencoder is trained on a model's intermediate activations to decompose them into an overcomplete, sparsely-activated dictionary of features. The idea (from Anthropic's mechanistic interpretability research) is that neural network activations are "superpositions" of many concepts packed into fewer dimensions. An SAE expands e.g. a 256-d bottleneck into 4096+ sparse features, where each feature ideally corresponds to one interpretable concept — a "monosemantic" unit. In our context, an SAE on the CNN bottleneck would ideally yield features like "TATA-box detector," "CpG-island indicator," or "AT-repeat scanner" that cleanly map onto biological motifs.

**What causal patching is:** Causal patching (activation patching) involves running the model on two different inputs, then surgically replacing an activation from one run into the other to see if the output flips. This identifies which internal components are *causally responsible* for a specific prediction — not just correlated, but necessary.

**Why we didn't implement either, in hindsight:** Both techniques require a model that has learned meaningful internal structure to decompose or patch. Our CNN achieved ~19% accuracy on 18 classes — it learned coarse family-level groupings but had very poor fine-grained discrimination. Linear probing on the bottleneck confirmed this: a family-level probe reached 63% but a fine-label probe could only reach 18.6% (no better than the model itself). SAE feature decomposition on such weak representations would yield noisy, uninterpretable features rather than clean monosemantic units. Causal patching would similarly be uninformative — you can't isolate "the circuit for state X" when the model barely distinguishes state X from most other states. The return on implementation effort was not justified given the model's performance floor.



### Phase 6: Activation Vector Steering (Inference-Time Intervention)
**Goal:** Improve predictions without retraining by intervening directly on internal representations at inference time. This is a form of "representation engineering" — if we know what direction in activation space corresponds to each class, we can nudge uncertain predictions along that direction.

**What activation vector steering is:**
1. **Extract activations:** Run all training sequences through the model and cache the bottleneck-layer activations (256-d vectors after global pooling).
2. **Compute label centroids:** For each of the 18 labels, compute \(\mu_k\) = mean activation vector across all training samples of label \(k\).
3. **Define steering vectors:** The vector from label \(A\) to label \(B\) is \(v_{A \to B} = \mu_B - \mu_A\). This is the direction in activation space that, in principle, shifts from "A-like" to "B-like."
4. **Apply at inference:** For a sample where the model is uncertain (confidence < threshold), add \(\alpha \cdot v_{pred \to target}\) to its bottleneck activation, then re-run only the classifier head. If steering toward a competing label increases confidence more than reinforcing the original prediction, flip the label.

**We ran three separate evaluation campaigns across two model versions.**

**Run 1 — Original ChromatinCNN (1024-d bottleneck):** Validation accuracy 17.7%, RC consistency 57.2%. Contrastive steering identified the top 10 confused pairs (label 0 ↔ 13 at 28.8%, label 8 ↔ 10 at 26.1%, etc.) and applied bidirectional correction at confidence threshold 0.6. Result: corrections were applied to 31.6% of samples, but 746 were hurt vs. 712 helped (16,629 neutral). Net improvement: **−0.06%** (slightly harmful).

**Run 2 — Clean model with family hierarchy (512-d bottleneck):** Validation accuracy 18.8%, family accuracy 63.6%, RC consistency 60.9%. This run tested directed steering with \(\alpha \in \{0, 0.25, 0.5, 0.75, 1.0, 1.5\}\) in three experiments:

*Experiment A — Within-family steering (label 0 → label 1, both promoter_polycomb, n=3,160):*

| \(\alpha\) | Fine Accuracy | Family Accuracy | Target Prob | Mean Confidence |
|:------:|:-------------:|:---------------:|:-----------:|:---------------:|
| 0.00   | 22.8%         | 65.4%           | 0.113       | 0.251           |
| 0.50   | 23.4%         | 64.3%           | 0.119       | 0.244           |
| 1.00   | 24.0%         | 63.0%           | 0.125       | 0.237           |
| 1.50   | 24.7%         | 61.6%           | 0.131       | 0.231           |

Fine accuracy rises +1.9pp but family accuracy drops −3.8pp — each unit of fine-label improvement costs ~2 units of family accuracy.

*Experiment B — Across-family steering (label 0 → label 12, promoter → enhancer, n=3,160):*

| \(\alpha\) | Fine Accuracy | Family Accuracy | Target Prob | Mean Confidence |
|:------:|:-------------:|:---------------:|:-----------:|:---------------:|
| 0.00   | 22.8%         | 65.4%           | 0.019       | 0.251           |
| 0.50   | 23.0%         | 63.0%           | 0.021       | 0.242           |
| 1.00   | 23.6%         | 60.2%           | 0.023       | 0.232           |
| 1.50   | 24.5%         | 57.7%           | 0.026       | 0.223           |

Across-family steering is even more destructive: family accuracy drops −7.7pp to push fine accuracy up +1.7pp. Target probability barely moves (0.019 → 0.026).

*Experiment C — Global steering toward label 14 (TssA, auto-source, n=56,722):*

| \(\alpha\) | Fine Accuracy | Family Accuracy | Target Prob | Mean Confidence |
|:------:|:-------------:|:---------------:|:-----------:|:---------------:|
| 0.00   | 18.2%         | 63.3%           | 0.055       | 0.159           |
| 0.50   | 18.2%         | 63.4%           | 0.059       | 0.157           |
| 1.00   | 17.9%         | 63.3%           | 0.063       | 0.156           |
| 1.50   | 17.6%         | 62.9%           | 0.068       | 0.156           |

On the full validation set, steering toward the model's best-separated class (TssA) actually *decreases* accuracy. Mean confidence is only ~16% across 18 classes — near uniform.

**Run 3 — Improved ChromatinCNNAttention (256-d bottleneck):** Validation accuracy 19.1%, RC consistency 78.1%. Contrastive steering on this model's top 10 confused pairs (label 0 ↔ 13 at 36.6%, label 14 ↔ 13 at 30.1%, label 17 ↔ 12 at 26.5%) produced: original accuracy 19.115%, corrected accuracy 19.113%. Net improvement: **−0.002%**.

**Representation geometry (from MI analysis):**

| Metric | Value |
|:-------|:-----:|
| Bottleneck dimension | 512 |
| Within-family centroid distance (mean) | 2.38 |
| Across-family centroid distance (mean) | 5.53 |
| Within/across ratio | 0.43 |
| Mistakes within-family | 50.5% |
| Mistakes across-family | 49.5% |
| Linear probe — family (3 classes) | 63.1% val |
| Linear probe — subcluster (7 classes) | 37.9% val |
| Linear probe — fine label (18 classes) | 18.6% val |

**How to interpret this:** The steering vectors are geometrically real — label centroids do separate in activation space (within-family distance ~2.4 vs. across-family ~5.5, ratio 0.43). But every steering experiment shows the same pattern: target probability nudges up slightly while overall accuracy stays flat or drops. The linear probes confirm why: the bottleneck's information ceiling is ~63% for 3-class family prediction and only ~18.6% for fine labels (no better than the model itself). Steering cannot conjure discriminative information that the representations never learned. The 50/50 within- vs. across-family mistake split further shows that the model's errors are not structured along biological lines — they are effectively random confusions, leaving no systematic pattern for steering to exploit.




### Phase 7: Diagnostics & Saliency

**Goal:** Verify whether the model uses biologically meaningful sequence features, and characterize its failure modes.

**Tools & Methods:**
- Input × Gradient saliency maps: compute gradient of a target class logit w.r.t. input nucleotides, then multiply element-wise by the input (highlights positions the model is "looking at")
- Confusion matrix analysis: identify which label pairs the model confuses most
- Aggregate saliency profiles per class: average saliency across many correctly-classified examples to see if position-level importance patterns differ by state

**Confusion analysis — top confused pairs (clean model, 0-indexed):**

| True Label | Predicted Label | Confusion Score |
|:----------:|:---------------:|:---------------:|
| 12         | 17              | 0.733           |
| 9          | 17              | 0.629           |
| 5          | 17              | 0.618           |
| 4          | 17              | 0.531           |
| 10         | 17              | 0.497           |

Many labels collapse into label 17 (enhancer_sub2), suggesting the model defaults to a dominant mode when uncertain. After adding hierarchy data, the improved model's confusions shifted to more biologically local pairs (e.g., label 0 ↔ 13, label 14 ↔ 13 — both within promoter_polycomb/background boundary).

**Alignment metrics:**
- Reverse-Compliment Strand consistency: 60.9% (clean) → 78.1% (improved) — meaning 22–39% of predictions flip on reverse complement, indicating fragile position-dependent features
- Monotonicity test (TATA box): score = 0.0 — strengthening a TATA motif did not increase promoter-label confidence at all, confirming the model has not learned to use known biological motifs
- Calibration ECE: 0.005 before / 0.010 after temperature scaling — trivially low because the model outputs near-uniform distributions (~16% mean confidence on 18 classes)

![confusion_matrix](file:///Users/xiaoyuwang/Desktop/manifold-dim-reduct/results/phase6_hierarchy/figures/confusion_matrix.png)

![pca_family](file:///Users/xiaoyuwang/Desktop/manifold-dim-reduct/results/phase6_hierarchy/figures/pca_family.png)

PCA of bottleneck activations colored by family assignment. Coarse family structure is visible, but fine-grained label separation is absent.





### Phase 8: Enriching Metadata — Label Identification & Hierarchy Construction

**Goal:** The competition labels are abstract integers (1–18) with no biological annotation. To make MI results interpretable and to enable biologically-informed steering, we needed to map each label to its likely chromatin state identity.

**Method:**

1. **Per-label feature profiling:** For each of the 18 labels, compute mean sequence-level features across ~6,000 stratified samples: GC content, CpG observed/expected ratio, CpG dinucleotide frequency, repeat density (homopolymer + dinucleotide tandem runs), and Shannon entropy.
2. **Expected state prototypes:** Define literature-based feature profiles for all 18 Roadmap Epigenomics states (e.g., TssA expects high GC ~0.65, high CpG ratio ~0.85; Het expects low GC ~0.35, high repeat density ~0.45).
3. **Anchor assignment:** Identify high-confidence anchors heuristically — the label with the highest CpG ratio is almost certainly TssA; the one with the lowest GC + highest repeats is Het; etc.
4. **Hungarian matching:** Z-score normalize observed label profiles and expected prototypes, compute Euclidean distance matrix, then use the Hungarian algorithm (scipy `linear_sum_assignment`) for optimal 1-to-1 matching of the remaining labels to states.
5. **Hierarchy construction:** Group the 18 matched states into 3 families (promoter_polycomb, enhancer_like, background_repressed) and 7 subclusters. Feed this hierarchy back into Phase 6 (family-level steering, family-level accuracy tracking) and Phase 7 (confusion analysis by family).

**Key identification results:**

| Label | Mapped State | GC Content | CpG Ratio | Repeat Density | Confidence Margin |
|:-----:|:------------:|:----------:|:---------:|:--------------:|:-----------------:|
| 14    | TssA         | 0.657      | 0.741     | 0.119          | 0.147             |
| 1     | ReprPC       | 0.582      | 0.683     | 0.116          | 0.264             |
| 2     | Het          | 0.517      | 0.551     | 0.137          | 0.556             |
| 13    | Quies        | 0.389      | 0.181     | 0.113          | 0.147             |
| 18    | ReprPCWk     | 0.387      | 0.188     | 0.111          | 0.157             |

Extreme states (TssA, Het, Quies) were identified with reasonable confidence. Middle states (various enhancer subtypes) had very small margins (<0.02), reflecting genuine biological similarity in their sequence composition.

**Family hierarchy output:**

```json
{
  "1":  {"family": "promoter_polycomb",    "subcluster": 1, "name_hint": "TssBiv_or_ReprPC_like"},
  "2":  {"family": "promoter_polycomb",    "subcluster": 2, "name_hint": "TssFlnk_like"},
  "3":  {"family": "enhancer_like",        "subcluster": 1, "name_hint": "enhancer_sub1"},
  "4":  {"family": "enhancer_like",        "subcluster": 1, "name_hint": "enhancer_sub1"},
  "5":  {"family": "background_repressed", "subcluster": 2, "name_hint": "background_sub2"},
  "6":  {"family": "background_repressed", "subcluster": 2, "name_hint": "background_sub2"},
  "7":  {"family": "enhancer_like",        "subcluster": 3, "name_hint": "enhancer_sub3"},
  "8":  {"family": "enhancer_like",        "subcluster": 3, "name_hint": "enhancer_sub3"},
  "9":  {"family": "enhancer_like",        "subcluster": 2, "name_hint": "enhancer_sub2"},
  "10": {"family": "background_repressed", "subcluster": 2, "name_hint": "background_sub2"},
  "11": {"family": "enhancer_like",        "subcluster": 2, "name_hint": "enhancer_sub2"},
  "12": {"family": "enhancer_like",        "subcluster": 3, "name_hint": "enhancer_sub3"},
  "13": {"family": "background_repressed", "subcluster": 1, "name_hint": "background_sub1"},
  "14": {"family": "promoter_polycomb",    "subcluster": 1, "name_hint": "TssA_like"},
  "15": {"family": "promoter_polycomb",    "subcluster": 1, "name_hint": "promoter_polycomb_sub1"},
  "16": {"family": "enhancer_like",        "subcluster": 3, "name_hint": "enhancer_sub3"},
  "17": {"family": "enhancer_like",        "subcluster": 2, "name_hint": "enhancer_sub2"},
  "18": {"family": "background_repressed", "subcluster": 1, "name_hint": "background_sub1"}
}
```

**Linear probing validated the hierarchy:** A linear probe on the model's bottleneck activations achieved 63.1% accuracy at the family level (3 classes) vs. only 18.6% at the fine label level (18 classes). This confirms the model's representations capture coarse biological structure but lack the resolution for fine-grained state discrimination. Confusion analysis showed 50.5% of errors within-family vs. 49.5% across-family — roughly chance-level, meaning the model's errors are not preferentially between biologically similar states.



### Conclusion

The model's poor performance (~19% accuracy, where random is 5.6%) was the dominant constraint on every MI experiment. Activation vector steering, confusion analysis, linear probing, and saliency maps all converge on the same finding: the CNN learns coarse family-level structure in its bottleneck (promoter-like vs. enhancer-like vs. background/repressed) but cannot resolve fine-grained distinctions within families. Steering vectors are geometrically real but act on representations too weak to move predictions meaningfully.

That said, MI was not wasted effort — it produced concrete, quantitative diagnostics that no amount of accuracy-chasing would have revealed:
- **Linear probing** established that the bottleneck's information ceiling is ~63% at the family level, directly explaining why fine-label accuracy plateaus near random
- **Steering analysis** confirmed the failure is in representation quality, not decision boundary placement — useful for deciding whether to invest in better features vs. bigger classifiers
- **Label identification + hierarchy** recovered biological structure from abstract competition labels, and the family-level PCA visualization validated that the manifold geometry from Phase 2 is echoed in learned representations
- **RC consistency and monotonicity** quantified specific failure modes (strand sensitivity, motif blindness) that inform architecture choices for future iterations

The broader takeaway: mechanistic interpretability on weak models yields "diagnostic interpretability" — it tells you *why* the model fails and *where* the information bottleneck lies, even if it cannot yet reveal meaningful learned circuits. For computational biology workflows, this diagnostic function may be as valuable as the circuit-level interpretability that MI achieves on stronger models.

