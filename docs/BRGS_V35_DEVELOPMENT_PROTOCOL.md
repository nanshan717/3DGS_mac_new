# BR-GS v3.5 Development Protocol

## Status and hypothesis

BR-GS v3.5 is an experimental successor to frozen v3.4. It preserves v3.4's
auditable ROI-consensus pruning and adds weak post-pruning recovery. The hypothesis
is that a small one-sided correction of low-opacity fabric Gaussians can recover
ground-truth accuracy without giving back v3.4's compactness gains.

Training does not read the ground-truth mesh. Ground truth is used only by
`eval_geometry_gt.py` after training.

## Recovery safeguards

- The Bernstein surface is frozen during recovery.
- Only post-pruning opacity in `[0.05, 0.20]` is eligible.
- The surface term is one-sided with a 3 mm dead zone.
- Each eligible centre stays within 5 mm of its post-pruning anchor.
- Recovery stops when reconstruction-loss EMA exceeds its pre-pruning reference by 3%.
- Densification remains stopped at pruning.
- Runs write `bsr_v34_pruning.json` and `bsr_v35_recovery.json`.

## Development split

P01 and P02 are development scenes because their v3.4 results were examined. Do not
tune v3.5 on P03/P04. After defaults are frozen, run P03/P04 once as held-out scenes,
then run the declared multi-seed evaluation.

## Smoke test

Normal pruning is at iteration 12000. For an 8000-iteration implementation smoke test
only, override it to 6000:

```bash
python train.py \
  -s /path/to/P01_flat_low_occlusion_v7_balanced96 \
  -m output/P01_v7_brgs_v35_smoke_8k \
  --eval -r 1 --iterations 8000 --seed 0 --deterministic \
  --bsr_v35 --bsr_prune_iter 6000
```

## Development comparison

Run the declared 15000-iteration configuration on P01 and P02 with seed 0:

```bash
python train.py \
  -s /path/to/SCENE \
  -m output/SCENE_brgs_v35_15k_s0 \
  --eval -r 1 --iterations 15000 --seed 0 --deterministic --bsr_v35
```

Use the same rendering and geometry evaluation scripts and thresholds as v3.4. Keep
official 3DGS and v3.4 outputs unchanged.

## Advance/freeze criterion

Advance v3.5 only if the P01/P02 aggregate improves ground-truth Chamfer L1 or
accuracy without a material rendering regression, while retaining meaningful point
reduction versus official 3DGS. Report all metrics; do not select presets per scene.
