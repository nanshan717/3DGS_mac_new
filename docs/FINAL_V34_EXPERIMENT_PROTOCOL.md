# BR-GS v3.4 final experiment protocol

## Decision and scope

BR-GS v3.4 is frozen as the paper's final method. No loss, threshold, pruning, surface,
ROI, densification, or recovery parameter may be changed after this point. BR-GS v3.5 is
a recorded negative/development ablation and is not advanced as the final method.

P01 and P02 are development scenes. The first formal task is a clean, matched three-seed
matrix on these scenes. P03/P04 will be generated and checksum-frozen only after this
matrix is complete; their results may not be used to tune v3.4.

The machine-readable declaration is `experiments/frozen_v34_matrix.json`.

## One-time official baseline preparation

Keep `/home/featurize/work/gaussian-splatting` pristine. Apply only the already required
16-bit PNG loader compatibility fix and the following RNG-control patch to the separate
compatibility checkout:

```bash
cd /home/featurize/work/3DGS_mac_new
conda activate megs

python tools/prepare_official_seed_support.py \
  /home/featurize/work/gaussian-splatting-p01-compat --apply
```

This writes `REPRODUCIBILITY_PATCH.json` in the compatibility checkout. The patch only
exposes `--seed` and `--deterministic` and replaces fixed seed 0 with the declared seed;
it does not alter the official algorithm. Preserve the audit file with the experiment
artifact. Never apply this helper to the pristine checkout.

## Run the frozen matrix

First print and inspect the commands:

```bash
python tools/run_frozen_v34_matrix.py
```

Run one cell at a time, which is safer on a remote server:

```bash
python tools/run_frozen_v34_matrix.py \
  --scene P01 --method official_3dgs --seed 0 --execute

python tools/run_frozen_v34_matrix.py \
  --scene P01 --method brgs_v34 --seed 0 --execute
```

Repeat the same two commands for seeds 1 and 2, then for P02. Each invocation performs
training, rendering, common `metrics.py` evaluation, and GT mesh geometry evaluation.
Training refuses to reuse an existing output directory by default. Do not add
`--allow_existing_train` for the final matrix.

For recovery after a completed training stage, select later stages explicitly:

```bash
python tools/run_frozen_v34_matrix.py \
  --scene P01 --method brgs_v34 --seed 0 \
  --stage render --stage metrics --stage geometry --execute
```

## Validate and aggregate

After all 12 cells exist:

```bash
python tools/aggregate_frozen_v34.py
cat comparisons/frozen_v34_final/summary.md
```

The aggregator fails closed on missing runs, source/iteration/seed mismatch, absent v3.4
audit files, point-count mismatch, or violation of the frozen 5% pruning cap. It writes:

- `per_run.csv`: every scene/method/seed result;
- `summary.csv`: mean and sample standard deviation;
- `summary.json`: complete machine-readable provenance and metrics;
- `summary.md`: paper-oriented table and change relative to official 3DGS.

## Reporting boundary

P01/P02 tables must be labeled development/multi-seed evidence. Report PSNR, SSIM,
LPIPS, point count, GT Accuracy, Completeness, Chamfer L1, F-score@1/2/5cm, and pruning
fraction. A favorable rendering/compactness trade-off does not justify claiming universal
geometry superiority when Accuracy or Chamfer is worse.
