    "train_sequences": "data/trainsequences.csv",
    "train_labels": "data/trainlabels.csv",
    "test_sequences": "data/testsequences.csv",

---

## Bug: Python Module Import Errors with `python -m`

**Date**: 2026-01-02

**Problem**:
When running `python -m phase3_model.run_phase3`, import errors occurred:
```
ModuleNotFoundError: No module named 'model'
ImportError: attempted relative import beyond top-level package
```

**Root Cause**:
- `run_phase3.py` adds parent directory to sys.path
- Python treats `phase3_model` as a package when running with `-m`
- Submodules (`train.py`, `inference.py`, etc.) used relative imports (`from .model`)
- Relative imports don't work when package is imported via `-m`

**Solution**:
Change all submodule imports from relative to package imports:

**Before (incorrect)**:
```python
# In train.py, inference.py, test_phase3.py
from .model import ChromatinCNN
from .dataset import ChromatinDataModule
```

**After (correct)**:
```python
# In train.py, inference.py, test_phase3.py
from phase3_model.model import ChromatinCNN
from phase3_model.dataset import ChromatinDataModule
```

**Files Modified**:
- `phase3_model/train.py`
- `phase3_model/inference.py`
- `phase3_model/test_phase3.py`

**Key Insight**:
When using `python -m package.module`, all submodules must use absolute package imports (`from package.module import X`), not relative imports (`from .module import X`). The parent directory in sys.path allows the package structure to be resolved correctly.

---

## Bug: OSError: [Errno 24] Too Many Open Files (MPS + Multiprocessing)

**Date**: 2026-01-02

**Problem**:
Training crashed with `OSError: [Errno 24] Too many open files` during validation dataloader.

**Root Cause**:
- macOS has a default limit on number of open file descriptors (typically 256)
- PyTorch DataLoader with `num_workers=4` spawns 4 worker processes
- Each worker process opens file descriptors for data loading
- With `pin_memory=True` and MPS device, additional file descriptors are used
- Total file descriptors exceed macOS limit of 256

**Solution**:

**Option 1: Reduce num_workers (Recommended)**
```json
{
  "phase3": {
    "training": {
      "num_workers": 2  // Reduce from 4 to 2
    }
  }
}
```

**Option 2: Disable pin_memory (if MPS issues persist)**
```python
# In train.py, ChromatinDataModule.__init__:
use_pin_memory = False  # Disable pinned memory
```

**Option 3: Disable data caching (if memory is an issue)**
```json
{
  "phase3": {
    "training": {
      "cache_data": false  // Reduce memory usage
    }
  }
}
```

**Option 4: Use spawn method (Mac-specific)**
```bash
# Set multiprocessing start method
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Recommended Fix**: Combine Options 1 + 2:
```json
{
  "phase3": {
    "training": {
      "num_workers": 2,
      "pin_memory": false,
      "cache_data": true
    }
  }
}
```

**Files to Modify**:
- `config.json`: Reduce `num_workers` to 2
- `phase3_model/dataset.py`: Change default `pin_memory=True` to `pin_memory=False`

**Temporary Command-Line Fix**:
```bash
# Run with reduced workers
nohup python -m phase3_model.run_phase3 \
    --config config.json \
    --mode both \
    --num_epochs 50 \
    --batch_size 256 \
    --device mps \
    > nohup_phase3_training.out 2>&1 &
```

**Note**: MPS (Metal Performance Shaders) on macOS has known issues with multiprocessing file descriptor limits. Reducing workers and/or disabling pin_memory resolves this.


Bug "OSError: [Errno 24] Too many open files". This error occurs when the DataLoader creates worker processes that don't properly clean up file descriptors between epochs.

solve: The DataLoader with multiprocessing workers is accumulating file descriptors across epochs. The problem is that workers aren't being properly cleaned up between training and validation phases. 