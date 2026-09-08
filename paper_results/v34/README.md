# BR-GS v3.4 paper-results package

This directory is generated from the checksum-frozen development and pre-registered
held-out matrices. It contains the complete 24-run data, scene summaries, paired-seed
deltas, a paper table, and two vector figures.

Source inputs:

- `comparisons/frozen_v34_final/per_run.csv` (P01--P02 development)
- `comparisons/heldout_v34_final/per_run.csv` (P03--P04 held-out)

Regenerate from the repository root with:

```bash
python3 tools/make_v34_paper_results.py
```

The generator verifies both input `SHA256SUMS` manifests before reading the data and
validates the full 4 scenes x 2 methods x 3 seeds design. `SHA256SUMS` in this directory
covers every generated artifact except itself.

Interpretation boundary: all comparisons are descriptive. Bold table entries mark only
the better per-scene mean and do not indicate statistical significance.
