This summary provides the necessary background and administrative details for a teammate joining the **Chromatin State Prediction Challenge** at the UCLA MiniHack1. Following the background is a revised master plan designed to handle the potentially abstract nature of the challenge labels.



### **Competition Introduction & Requirements**

- **The Task**: Predict the correct **ChromHMM chromatin state annotation** for a given **200bp DNA sequence** in isolation.

- **Data Files**:

  - 

    **`trainsequences.csv`**: 286,164 DNA sequences (200bp each).

  - 

    **`trainlabels.csv`**: 286,164 labels corresponding to the training sequences, provided as integers from **1 to 18**.

  - 

    **`testsequences.csv`**: 100,008 sequences for which you must predict labels.

- **Dataset Balance**: Every one of the 18 states is represented in **equal proportions** (1/18th of the data) in both the training and test set.

- **Key Rules**:

  - **No External Data**: You may only use the provided files.

  - **No Specialized Software**: You cannot use existing software specifically designed for DNA sequence prediction8. Standard machine learning libraries (e.g., PyTorch, Scikit-learn) are allowed.

    

- **Submission Details**:

  - **Format**: A file named **`predictions.csv`** containing 100,008 integers (one per line).

  - **Packaging**: The `predictions.csv` file **must be placed in a zip file** (any name) for upload.

  - **Platform**: Submissions are uploaded to **Codabench**.

    

### **The 6 Underlying Histone Marks**

The 18 states are defined by the presence or absence of specific chemical "marks" on the DNA packaging proteins. The model for this challenge is based on these **six marks**:

1. **H3K4me3**: Active transcription start sites.
2. **H3K27ac**: Active enhancers and promoters.
3. **H3K4me1**: Enhancer regions.
4. **H3K36me3**: Actively transcribed gene bodies.
5. **H3K9me3**: Repressed/silenced heterochromatin.
6. **H3K27me3**: Polycomb-repressed regions.



### **Master Plan: "The Discovery Engine"**

Because the 18 labels in the CSV may be abstract (integers 1–18) and potentially divorced from their original biological meanings in the PDF, the plan shifts from "label prediction" to **"functional discovery."**

### Phase 1: Genomic Data Engineering & Augmentation

The goal is to bake biological "first principles" into your dataset before the model even sees it.

- **Strand Invariance (RC Augmentation):** Double your training set by generating the **Reverse Complement (RC)**for every 200bp sequence111. This ensures the model treats `GATTACA` and `TGTAATC` as biologically identical.
- **Position Jittering:** During training, randomly crop 180bp windows from the 200bp sequences. This forces the model to recognize motifs regardless of their location within the interval.

### Phase 2: Exploratory Manifold Analysis

Map the "functional landscape" of the genome to understand how the 18 states overlap.

- 

  **K-mer Vectorization:** Convert the sequences into 5-mer or 6-mer frequency counts to capture the "local vocabulary".

  

  

- 

  **PHATE/UMAP Embedding:** Use **PHATE** to visualize the 286,164 training samples in a low-dimensional space.

  

  

- **Goal**: Determine which abstract labels (e.g., Label 3 vs Label 17) are biologically similar and which are distinct. This prevents the model from treating "Active" and "Repressed" states as equally similar.

### Phase 3: The Mechanistic "Engine" (Architecture)

Build a model designed for transparency rather than complexity.

- **1D-CNN Core:** Use a 1D Convolutional layer with 32–64 filters of length 15–20bp. These filters will act as automated scanners for biological motifs.
- **Global Max Pooling:** Apply this after the CNN layer to achieve **position invariance**—extracting the single strongest "firing" of a motif anywhere in the sequence9.
- **Sparse Autoencoder (SAE) Attachment:** Attach a Sparse Autoencoder to the CNN activations. The SAE will "de-mix" the overlapping signals of the CNN into clean, **monosemantic motifs** (e.g., a "pure" TATA-box feature).
- **Mapping Meaning**: If SAE Feature A activates for Label 1, and Feature A represents a known promoter motif, we can conclude that **Label 1 = Promoter-like**.

### Phase 4: Discovery & Causal Patching

Use Mechanistic Interpretability to "decompile" what the model has learned.

- **Circuit Tracing**: Take a sequence the model labels as "13" and "patch" in the activations of a "Label 1" motif.
- **Logic**: If the model output flips to "1," you have identified the **causal DNA code** for that abstract label.

### Phase 5: Steering for Performance (Inference-Time Intervention)

Use your discovery of biological patterns to boost accuracy on the 100,008 test sequences.

- **Steering Vectors**: Calculate the "Average Activation Direction" for each of the 18 states.
- **Inference Correction**: If the model is unsure between two states, project the activations onto your discovered "State Vectors." Use the steering nudge to resolve the tie toward the state that matches the underlying biological motifs.

### Summary of Competition Specs

- **Input:** 200bp DNA sequences (A, C, G, T only).

- **Task:** Predict one of 18 functional labels15.

- **Data Balance:** Each state is present in equal proportions in training and test sets16161616.

- **Rule Reminder:** No external data is allowed; you must rely entirely on the provided `trainsequences.csv`17.