"""Phase 7: model diagnostics.

Three independent probes, all single-GPU:

- `saliency.py`    — SmoothGrad-based per-base importance maps.
- `calibration.py` — temperature scaling + expected calibration error.
- `consistency.py` — reverse-complement prediction consistency.

`run_phase7.py` runs all three and emits `results/phase7_diagnostics/`
artifacts consumed by phase 8 and the final writeup.
"""

from .saliency import smoothgrad_saliency
from .calibration import expected_calibration_error, fit_temperature
from .consistency import reverse_complement_consistency

__all__ = [
    "smoothgrad_saliency",
    "expected_calibration_error",
    "fit_temperature",
    "reverse_complement_consistency",
]
