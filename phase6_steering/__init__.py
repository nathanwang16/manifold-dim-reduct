"""Phase 6: Representation Engineering & Steering on the bottleneck.

Steering vectors are computed as (class centroid − global centroid) on the
384-d bottleneck representation of a trained `ChromatinCNNAttentionV2`.
They are then added (with scalar alpha) to the bottleneck during inference
to push predictions toward a target class — a quick diagnostic of whether
the learned representation is "steerable" and a regulariser for confused
class pairs.

The whole phase is a single-GPU evaluation (no training). The heavy
lifting is done in `steering.py`; `run_phase6.py` is the orchestrator.
"""

from .steering import (
    SteeringVectors,
    compute_steering_vectors,
    steered_forward,
)

__all__ = [
    "SteeringVectors",
    "compute_steering_vectors",
    "steered_forward",
]
