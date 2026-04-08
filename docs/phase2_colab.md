# Phase 2: Manifold Learning - Google Colab Instructions

This notebook (`phase2_colab.ipynb`) is a complete, self-contained implementation of Phase 2 of the manifold learning pipeline, designed to run on Google Colab with GPU/CPU support.

## 🚀 Quick Start

### Option 1: Open Directly in Google Colab
1. Upload `phase2_colab.ipynb` to your Google Drive
2. Open [Google Colab](https://colab.research.google.com/)
3. File → Open notebook → Upload → Select `phase2_colab.ipynb`
4. Run cells sequentially or click "Runtime → Run all"

### Option 2: From GitHub/GitLab
1. Copy the notebook URL to your repository
2. In Colab: File → Open notebook → GitHub
3. Paste URL and select the notebook

## 📁 Data Preparation

### Method 1: Upload Your Own Data (Recommended)

1. **Prepare your CSV files**:
   - `train_sequences.csv`: DNA sequences (one sequence per row, NO header)
   - `train_labels.csv`: Labels (one label per row, NO header)

   **Example format for `train_sequences.csv`**:
   ```
   ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
   TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC
   ...
   ```

   **Example format for `train_labels.csv`**:
   ```
   1
   5
   12
   ...
   ```

2. **Upload to Google Drive**:
   - Create a folder in Google Drive named `chromatin_data`
   - Upload your CSV files to this folder
   - The folder structure should look like:
     ```
     My Drive/
       ├─ chromatin_data/
       │   ├─ train_sequences.csv
       │   └─ train_labels.csv
       ```

3. **Configure the notebook**:
   - In the "Configuration" cell (Cell 7), change:
     ```python
     USE_DEMO_DATA = False  # Set to False
     ```

### Method 2: Use Demo Data (For Testing)
- Default setting uses synthetic demo data (10,000 sequences)
- No file uploads needed
- Perfect for:
  - Testing the pipeline
  - Debugging
  - Understanding the workflow

## ⚙️ Configuration Options

The notebook includes configurable parameters in the Configuration cell (Cell 7):

### Data Settings
- `USE_DEMO_DATA`: Use synthetic data (True) or your uploaded data (False)
- `DATA_PATH`: Path to your data folder in Google Drive
- `OUTPUT_PATH`: Path for saving results
- `SUBSAMPLE_SIZE`: Limit number of sequences for faster testing (None = all data)

### Feature Extraction
- `KMER_K`: K-mer size (5 = 1024 features, 6 = 4096 features)
- `N_POSITIONAL_BINS`: Number of position bins (default: 10)
- `FEATURE_EXTRACTION_BATCH_SIZE`: Batch size for parallel processing (default: 1000)

### Dimensionality Reduction
- `PCA_N_COMPONENTS`: Number of PCA components (default: 50)
- `USE_PCA_PREPROCESSING`: Use PCA before PHATE for speed (recommended: True)
- `N_PCS_FOR_PHATE`: PCA components to keep for PHATE (default: 30)

### Parallel Processing
- `N_JOBS`: Number of parallel jobs (-1 = all cores, 2-4 = specific number)
  - Use higher values (e.g., 4-8) for Colab with GPU
  - Use -1 or 2-4 for CPU-only environments

## 💾 Output Files

All results are saved to your Google Drive in the `phase2_results` folder:

### Features
- `kmer_5_features.npy`: K-mer frequency matrix (n_samples × 1024)
- `positional_kmer_features.npy`: Positional k-mer profiles (n_samples × 10240)
- `dinucleotide_features.npy`: Dinucleotide frequencies (n_samples × 16)
- `labels.npy`: Label array (n_samples,)
- `kmer_5_vocab.json`: K-mer vocabulary mapping

### Embeddings
- `pca_2d.npy` / `pca_embeddings.npy`: PCA embeddings
- `umap_n15_d0.1.npy`, etc.: UMAP embeddings with different parameters
- `phate_k10_d20_pca30.npy`, etc.: PHATE embeddings with different parameters
- `pca_variance_ratio.npy`: PCA variance explained per component

### Analysis
- `analysis/cluster_analysis_results.json`: Full analysis results including:
  - Silhouette scores (overall and per-class)
  - Adjusted Rand Index values
  - Label centroids
  - Predicted confusion pairs
  - Comparison summary

### Visualizations
- `figures/pca_2d.png`: PCA 2D scatter plot
- `figures/umap_n15_d0.1_2d.png`, etc.: UMAP scatter plots
- `figures/phate_*.png`: PHATE scatter plots
- `figures/pca_variance_explained.png`: PCA variance plot
- `figures/method_comparison_bars.png`: Method comparison chart

### Download
- `phase2_results.zip`: All results compressed (created automatically)

## ⏱️ Expected Runtime

Approximate times on Google Colab:

| Dataset Size | Feature Extraction | PCA | UMAP (4 configs) | PHATE (3 configs) | Total |
|--------------|-------------------|------|-------------------|---------------------|--------|
| 10K (demo) | 2-3 min | 1 min | 5-10 min | 3-5 min | 12-20 min |
| 50K | 10-15 min | 3-5 min | 20-30 min | 15-25 min | 50-75 min |
| 100K | 20-30 min | 8-12 min | 40-60 min | 30-45 min | 100-150 min |
| 170K (full) | 35-45 min | 15-20 min | 60-90 min | 50-70 min | 160-225 min |

**Tips to reduce runtime:**
- Use `SUBSAMPLE_SIZE` to limit sequences (e.g., 50000)
- Reduce `KMER_K` from 5 to 4 (fewer features = faster processing)
- Reduce number of UMAP/PHATE parameter combinations
- Use GPU runtime (accelerates UMAP significantly)

## 🔧 Troubleshooting

### "Sequence file not found"
**Solution**: Ensure CSV files are uploaded to `chromatin_data` folder in Google Drive and path is correct.

### Out of Memory (OOM)
**Solutions**:
1. Reduce `SUBSAMPLE_SIZE` to 10000-50000
2. Set `KMER_K = 4` (fewer features)
3. Use `N_JOBS = 2` instead of -1
4. Restart runtime and clear cache: Runtime → Restart runtime

### UMAP/PHATE too slow
**Solutions**:
1. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
2. Use PCA preprocessing (already enabled by default)
3. Reduce number of parameter combinations
4. Use subsampling

### Visualization not displaying
**Solution**: Colab displays plots inline automatically. If not showing, check:
- `%matplotlib inline` should be active
- Scroll down in output cell
- Try creating a new cell with: `plt.show()`

### Phase 2 results for Phase 3
**Key files to download**:
1. `kmer_5_features.npy` - Use as input for CNN model
2. `labels.npy` - Labels for supervised learning
3. `analysis/cluster_analysis_results.json` - Analysis insights

## 📊 Interpreting Results

### Silhouette Score
- Range: -1 to 1
- **> 0.5**: Well-clustered, distinct classes
- **0.25-0.5**: Reasonable separation
- **< 0.25**: Overlapping clusters, may need better features

### Adjusted Rand Index (ARI)
- Range: -1 to 1
- **> 0.5**: Good agreement between clusters and true labels
- **0.3-0.5**: Moderate agreement
- **< 0.3**: Poor clustering quality

### Best Method Selection
Look for method with:
- Highest silhouette score
- Highest ARI
- Visual separation in 2D plot

Typical results (in order of performance):
1. UMAP (n_neighbors=15, min_dist=0.1) - often best for local structure
2. PHATE - good for global/trend structure
3. PCA - baseline linear method

## 🎯 Best Practices

1. **Start with demo data** to verify pipeline works
2. **Use subsampling** for initial exploration
3. **Save intermediate results** (automatically saved to Drive)
4. **Monitor GPU usage** in Colab (Runtime → Show code metrics)
5. **Check visualizations** before proceeding to next phase
6. **Document findings** from cluster analysis

## 📚 Additional Resources

- **Methodology**: See `guide.md` in the repository for detailed Phase 2 methodology
- **Local execution**: See original Phase 2 Python scripts in this folder
- **Dependencies**: All required packages are installed automatically
- **Data format**: Ensure CSV files have NO header row

## 🆘️ Getting Help

1. **Check log output** - Each section prints detailed progress
2. **Review visualizations** - Examine generated plots for issues
3. **Compare with demo** - Run demo data to establish baseline
4. **Consult guide.md** - Detailed methodology in repository
5. **Check Colab resources** - Ensure sufficient RAM/CPU allocated

## ✅ Success Criteria

You have successfully completed Phase 2 when:
- ✅ Features extracted without errors
- ✅ All embeddings generated (PCA, UMAP, PHATE)
- ✅ Cluster analysis completed with silhouette and ARI scores
- ✅ Visualizations generated and saved
- ✅ Results saved to Google Drive
- ✅ Zip file created for download

## 📝 Notes

- All computations are performed on Colab's cloud resources
- Results persist in Google Drive even after session ends
- You can resume by running notebook again (if using same output path)
- GPU runtime is recommended for large datasets (>50K sequences)

---

**Last Updated**: January 2026
**Version**: 1.0








