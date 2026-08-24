# BR-GS v3 experiment protocol

## Compatible workflow

The existing conversion, rendering, and metric commands remain valid. Vanilla:

```bash
python train.py -s /path/to/scene -m output/scene_3dgs --eval -r 4
```

Recommended target-domain BR-GS v3:

```bash
python train.py -s /path/to/scene -m output/scene_brgs_v3 --eval -r 4 --bsr_v3
python render.py -m output/scene_brgs_v3
python metrics.py -m output/scene_brgs_v3
python eval_geometry.py -m output/scene_brgs_v3 --axis auto --save_json
```

For annotated fabric regions, store masks using the source image filename or stem under
`<scene>/bsr_masks/`, then add:

```bash
--bsr_roi_dir bsr_masks --bsr_roi_required
```

Every run writes `experiment_manifest.json` with resolved dataset, optimizer, pipeline,
software, and command-line settings. Do not compare runs unless their dataset split,
resolution, seed, and evaluation settings match.

## Frozen ablation order

Run at least three seeds. Select hyperparameters on development scenes only and freeze
them before evaluating held-out scenes.

| ID | Configuration |
|---|---|
| A0 | Vanilla 3DGS |
| A1 | Legacy BR-GS (`--use_bsr`) |
| A2 | A1 + automatic support axis |
| A3 | A2 + normalized robust distance |
| A4 | A3 + height-only controls |
| A5 | A4 + spatial sampling |
| A6 | A5 + bidirectional coverage |
| A7 | A6 + floater opacity loss |
| A8 | A7 + piecewise surface and seam continuity |
| A9 | A8 + ROI masks |
| Full | `--bsr_v3`, plus ROI when available |

`--bsr_v3` currently resolves to a 2x2 patch grid, height-only controls, automatic
support direction, normalized Huber distance, density-aware weights, bidirectional
coverage, control smoothness, seam continuity, spatial sampling, floater opacity loss,
and delayed scheduling. The manifest, not this document, is the authority for an actual
run.

## Reporting

Report PSNR, SSIM, LPIPS, training time, peak memory, point count, and geometry metrics.
The legacy spacing-based floater ratio is retained only for reproduction. Primary
geometry reporting should use normalized GSD, P50/P90/P95 distance, floater ratios at
1/2/5 percent of support-region span, local surface variation, and—when ground truth is
available—Chamfer distance, F-score, normal consistency, and fabric-only depth RMSE.

