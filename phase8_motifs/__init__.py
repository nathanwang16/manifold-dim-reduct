"""Phase 8: Motif discovery & hypothesis generation.

With meta-derived ground-truth state names the previous "label
identification" objective is moot. This phase instead uses the
trained phase-3 model as a microscope on chromatin sequence patterns.

Three complementary probes, all single-GPU:

- `stem_motifs.py`  — extract PWMs from the 256 stem Conv1d filters by
                      accumulating top-activating 19-bp windows on val
                      sequences. Writes per-filter PWM arrays + class
                      association (which classes does this filter help?).
- `ism.py`          — in-silico mutagenesis: per-base, per-substitution
                      logit deltas for the top-activating examples of
                      each class. This produces per-position "impact
                      scores" that nominate candidate motifs / sequence
                      contexts.
- `hypotheses.py`   — consolidates PWM + saliency + ISM results into
                      testable hypothesis records (motif → associated
                      class(es) → candidate biological interpretation).

`run_phase8.py` runs all of the above.
"""

from .stem_motifs import extract_stem_pwms
from .ism import in_silico_mutagenesis

__all__ = ["extract_stem_pwms", "in_silico_mutagenesis"]
