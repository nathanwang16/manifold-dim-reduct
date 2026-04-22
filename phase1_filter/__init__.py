"""Phase 1: Data engineering for the roadmap_18state_full corpus.

The heavy lifting (liftover, balancing, RC augmentation for minority states,
family/subcluster metadata) is already done in `phase0_aggregate/`. Phase 1
now consists of thin post-processors:

- `extract_hierarchy_labels.py` — derive one-column family / subcluster label
  CSVs from the meta files, matching what the legacy training code expects.
- `build_subsamples.py` — emit stratified row-index arrays (`.npy`) for
  balanced visualization / MI / eval subsets from val and test.
- `filter_by_eid.py` — build index arrays restricted to a set of epigenome IDs
  (tissue/cell-type analyses in later phases).

Run `python phase1_filter/run_phase1.py` to execute all three with defaults.
"""
