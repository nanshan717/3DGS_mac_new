# BR-GS v3.4 frozen validation protocol

## Status

BR-GS v3.4 is the frozen final algorithm version for the paper. The implementation is
enabled only by `--bsr_v34`; do not tune its defaults against P01, P02, or any future
held-out test scene. BR-GS v3.5 is retained only as a negative/development ablation.

P01 V7 (`P01_flat_low_occlusion_v7_balanced96`) is a **development scene** because its
12 evaluation views were repeatedly inspected while developing v3.1--v3.4. Its results
are evidence that the method is viable, not an unbiased final benchmark result.

P02 V7 (`P02_undulating_medium_occlusion_v7_balanced96`) is also a **development scene**
because its results were inspected before the final-version decision. Neither P01 nor P02
may be relabeled as held-out performance.

## Frozen defaults

The resolved values in `experiment_manifest.json` are authoritative. The principal
v3.4 values are:

| Parameter | Value |
|---|---:|
| Training iterations | 15000 |
| Pruning iteration | 12000 |
| Recovery iterations | 3000 |
| Maximum pruning fraction | 0.05 |
| Maximum candidate opacity | 0.05 |
| Minimum normalized surface distance | 0.02 |
| Minimum ROI consensus | 0.60 |
| ROI views | 8 |
| Minimum valid ROI views | 2 |
| Surface patches | 2 x 2 |
| BSR warmup / ramp | 5000 / 5000 |

At pruning, a Gaussian must satisfy the opacity, surface-distance, and multi-view ROI
conditions simultaneously. Every run writes `bsr_v34_pruning.json`. Densification and
BR-GS regularization stop at pruning; the remaining iterations are photometric recovery.

## P01 development evidence (seed 0)

The official 3DGS checkout received only the documented `np.byte` to `np.uint8` image
loader compatibility change. All metrics were recomputed with the same scripts and
shared geometry support frame.

| Method | PSNR | SSIM | LPIPS | Points | Norm. GSD | P90 | Floater 5% | Roughness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official 3DGS | 18.930237 | 0.795055 | 0.374775 | 611773 | 0.011440 | 0.132241 | 0.014245 | 0.852870 |
| BR-GS v3.4 | 19.898256 | 0.797129 | 0.369148 | 448773 | 0.011005 | 0.116785 | 0.007164 | 0.848161 |

These values must not be presented as final held-out dataset performance. They may be
reported as a development study or ablation with that designation.

## Required final validation

1. Complete the declared P01/P02 three-seed development matrix without further tuning
   (completed: 12/12 validated runs).
2. Generate the pre-registered held-out P03/P04 scenes with seeds 303/404 and new geometry.
3. Freeze P03/P04 manifests, images, masks, camera splits, GT meshes, and checksums before
   any baseline or BR-GS result is inspected.
4. Run official 3DGS and BR-GS v3.4 with seeds 0, 1, and 2.
5. Report mean and standard deviation for image, geometry, point-count, time, and memory
   metrics; retain per-run JSON files.
6. Include official 3DGS, 2DGS, and at least one geometry-focused Gaussian baseline.
7. Evaluate real coffee-field captures separately and label synthetic and real results.

The executable P03/P04 freeze and run procedure is documented in
`docs/HELDOUT_V34_EXPERIMENT_PROTOCOL.md`.

## Seeded BR-GS command

```bash
python train.py \
  -s /absolute/path/to/scene \
  -m output/SCENE_brgs_v34_15k_sSEED \
  --eval -r 1 --iterations 15000 \
  --seed SEED --deterministic --bsr_v34
```

Never choose a checkpoint, pruning threshold, scene subset, or test-view subset based on
held-out test metrics. Failed and excluded runs must remain documented with their reason.
