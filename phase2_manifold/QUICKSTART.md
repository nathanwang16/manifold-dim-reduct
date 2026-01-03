# Phase 2 Colab Notebook - Quick Start Guide

## 📋 What You Have

Created for you in `phase2_manifold/`:

### 1. **phase2_colab.ipynb** (Main Deliverable)
A complete, self-contained Jupyter notebook ready to run on Google Colab with:
- 27 cells covering full Phase 2 pipeline
- Google Drive integration
- GPU/CPU detection and utilization
- Interactive configuration options with Colab forms
- All necessary imports and functions
- Automatic result saving and ZIP creation

### 2. **README_COLAB.md**
Comprehensive documentation including:
- Detailed setup instructions
- Data upload guide
- Configuration options
- Runtime estimates
- Troubleshooting guide
- Results interpretation

### 3. **QUICKSTART.md** (This file)
Quick reference for getting started

## 🚀 3 Steps to Run Phase 2 on Colab

### Step 1: Prepare Data (2 options)

**Option A - Use Your Data**:
1. Create folder `chromatin_data` in Google Drive
2. Upload `train_sequences.csv` and `train_labels.csv` to that folder
3. No headers in CSV files, one sequence/label per row

**Option B - Use Demo Data** (easiest for testing):
- Just set `USE_DEMO_DATA = True` in configuration cell
- No uploads needed

### Step 2: Open Notebook in Colab

**Method 1 - Direct Upload**:
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. File → Open notebook → Upload
3. Select `phase2_colab.ipynb` from this folder

**Method 2 - GitHub** (if pushed to repo):
1. File → Open notebook → GitHub
2. Paste your repository URL
3. Select the notebook

### Step 3: Run Pipeline

1. **Mount Google Drive** (Cell 6) - Click when prompted
2. **Configure settings** (Cell 7) - Adjust parameters as needed:
   - Set `USE_DEMO_DATA = False` if using your data
   - Adjust `SUBSAMPLE_SIZE` if needed (e.g., 50000 for faster testing)
3. **Run all cells**: Runtime → Run all

That's it! Results will be automatically saved to Google Drive.

## 📊 What You'll Get

After ~15-180 minutes (depending on dataset size), you'll have:

### Features
- K-mer frequency matrix (1024 features per sequence)
- Positional k-mer profiles (10240 features per sequence)
- Dinucleotide transitions (16 features per sequence)

### Embeddings (Multiple Methods)
- **PCA**: Linear baseline (50 components)
- **UMAP**: 4 parameter configurations (n_neighbors=15/30/50, min_dist=0.0/0.1/0.25)
- **PHATE**: 3 parameter configurations (knn=5/10/15, decay=10/20/40)

### Analysis
- Silhouette scores for each method
- Adjusted Rand Index (ARI) comparing clusters to true labels
- Per-class statistics (which classes cluster well/poorly)
- Predicted confusion pairs (which classes are most likely confused)

### Visualizations
- 2D scatter plots for each embedding method
- PCA variance explained plot
- Method comparison bar chart
- All saved as high-resolution PNG files

### Files for Next Phase (Phase 3 - Model Training)
- `kmer_5_features.npy`: Feature matrix for CNN input
- `labels.npy`: Label array for supervised learning
- `analysis/cluster_analysis_results.json`: Insights to guide model architecture

## ⚙️ Key Configuration Options

| Parameter | Default | Recommendation | Notes |
|-----------|---------|----------------|--------|
| `USE_DEMO_DATA` | True | False for your data | True = synthetic 10K sequences |
| `SUBSAMPLE_SIZE` | None | 50000 for testing | None = all data |
| `KMER_K` | 5 | Keep as 5 | 4 = faster but less detail |
| `N_JOBS` | -1 | -1 (all cores) | 2-4 if memory issues |
| `USE_PCA_PREPROCESSING` | True | True | Speeds up PHATE significantly |

## ⏱️ Time Estimates

| Dataset | Demo (10K) | Subsample (50K) | Full (170K) |
|----------|--------------|------------------|---------------|
| Feature Extraction | 2-3 min | 10-15 min | 35-45 min |
| PCA | 1 min | 3-5 min | 8-12 min |
| UMAP (4 configs) | 5-10 min | 20-30 min | 60-90 min |
| PHATE (3 configs) | 3-5 min | 15-25 min | 50-70 min |
| Analysis & Viz | 5 min | 10 min | 15-20 min |
| **Total** | **12-20 min** | **50-75 min** | **160-225 min** |

💡 **Tip**: Start with demo data (2-20 min) to verify everything works!

## 🐛 Common Issues & Fixes

### Issue: "Sequence file not found"
**Fix**: Ensure files are in `chromatin_data` folder in Google Drive, not in a subfolder

### Issue: Out of memory
**Fixes**:
1. Set `SUBSAMPLE_SIZE = 10000` in configuration
2. Change runtime to GPU (Runtime → Change runtime type)
3. Restart runtime and clear variables

### Issue: Too slow
**Fixes**:
1. Use GPU runtime (UMAP is much faster)
2. Reduce `SUBSAMPLE_SIZE`
3. Reduce number of UMAP/PHATE configs (edit `UMAP_PARAMS` and `PHATE_PARAMS` lists)

### Issue: Visualization not showing
**Fix**: Plots display inline in Colab. Scroll down in output cell. Still not showing? Add new cell:
```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.show()
```

## 📦 After Completion: Download Results

### Automatic ZIP
Notebook automatically creates `phase2_results.zip` in Google Drive.

### Manual Download
1. Open Google Drive in browser
2. Navigate to `phase2_results/`
3. Download individual files or the zip

### Key Files for Phase 3
Keep these for model training:
- `kmer_5_features.npy` - Features (n_samples × 1024)
- `labels.npy` - Labels (n_samples,)
- `pca_embeddings.npy` - Optional PCA features
- Best embedding file (e.g., `umap_n15_d0.1.npy`)

## 🎯 Next Steps

After Phase 2 completes successfully:

### For Phase 3 (CNN Training):
- Use `kmer_5_features.npy` as model input
- Consider silhouette scores to weight losses for difficult classes
- Use hierarchical clustering results to guide model architecture
- Refer to confusion predictions for data augmentation

### For Analysis:
- Review which embedding method performed best
- Identify well-clustered vs. dispersed classes
- Investigate biological patterns in hierarchical clustering
- Compare predictions with true confusion matrices later

## 📚 Additional Resources

- **Detailed guide**: See `README_COLAB.md` for comprehensive documentation
- **Methodology**: See `guide.md` in repository root
- **Local execution**: Original Python scripts are in this folder for local runs
- **Phase 3**: See `phase3_colab_training.ipynb` for next steps

## ✅ Checklist

Before starting:
- [ ] Have data files ready (or use demo data)
- [ ] Google Drive accessible
- [ ] Have ~30 minutes free (for demo) or ~3 hours (for full dataset)

After running:
- [ ] All cells executed without errors
- [ ] Features saved (check `phase2_results/` folder)
- [ ] Embeddings generated for all methods
- [ ] Visualizations created
- [ ] Analysis results saved
- [ ] ZIP file created for download

---

**Ready to run Phase 2?** Open `phase2_colab.ipynb` in Google Colab and get started! 🚀

