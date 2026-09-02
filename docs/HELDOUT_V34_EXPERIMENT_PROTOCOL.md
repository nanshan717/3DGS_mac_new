# BR-GS v3.4 P03/P04 held-out experiment protocol

## Scientific boundary

P01/P02 are development evidence. P03/P04 are new scene-level held-out tests. BR-GS v3.4
and its thresholds remain unchanged. Do not inspect any official-3DGS or BR-GS output on
P03/P04 until both datasets are complete, structurally validated, checksum-frozen, and the
freeze commit has been recorded. Input images may be checked for renderer failures before
freezing; after freezing, scene geometry, cameras, masks, splits and images are immutable.

Pre-registered designs:

| Scene | Dataset seed | Geometry | Occlusion | Lighting |
|---|---:|---|---|---|
| P03 | 303 | cross-slope with transverse wrinkles | high | soft/overcast |
| P04 | 404 | piecewise crossfall, local uplift and edge sag | mixed | oblique hard shadow |

Both use the V7 96-view paired camera protocol (84 train, 12 test), the same recorded
CC-BY-4.0 coffee asset, 800 x 800 RGB, and 800 x 800 visible-ground ROI masks.

## 1. Generate complete datasets on the Mac

Run from the repository root. These are data-generation seeds, distinct from the model
training seeds 0/1/2.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python tools/coffee_fabric_syn/generate_blender_scene.py -- \
  --output data/CoffeeFabric-Syn/heldout --seed 303 \
  --scene-profile p03 --dataset-version v7 --views 96 \
  --coffee-asset assets/coffee/a_coffee_tree/a_coffee_tree.glb \
  --asset-license CC-BY-4.0 --engine CYCLES \
  --preview-resolution 800 --preview-samples 48 \
  --render-dataset --dataset-resolution 800 --dataset-samples 48

/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python tools/coffee_fabric_syn/generate_blender_scene.py -- \
  --output data/CoffeeFabric-Syn/heldout --seed 404 \
  --scene-profile p04 --dataset-version v7 --views 96 \
  --coffee-asset assets/coffee/a_coffee_tree/a_coffee_tree.glb \
  --asset-license CC-BY-4.0 --engine CYCLES \
  --preview-resolution 800 --preview-samples 48 \
  --render-dataset --dataset-resolution 800 --dataset-samples 48
```

## 2. Render ROI masks and validate

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  data/CoffeeFabric-Syn/heldout/P03_sloped_high_occlusion_v7_heldout96/P03_sloped_high_occlusion_v7_heldout96.blend \
  --background --python tools/coffee_fabric_syn/render_ground_roi_masks.py -- \
  data/CoffeeFabric-Syn/heldout/P03_sloped_high_occlusion_v7_heldout96 --resolution 800

/Applications/Blender.app/Contents/MacOS/Blender \
  data/CoffeeFabric-Syn/heldout/P04_piecewise_mixed_occlusion_v7_heldout96/P04_piecewise_mixed_occlusion_v7_heldout96.blend \
  --background --python tools/coffee_fabric_syn/render_ground_roi_masks.py -- \
  data/CoffeeFabric-Syn/heldout/P04_piecewise_mixed_occlusion_v7_heldout96 --resolution 800

python tools/coffee_fabric_syn/validate_trainable_dataset.py \
  data/CoffeeFabric-Syn/heldout/P03_sloped_high_occlusion_v7_heldout96
python tools/coffee_fabric_syn/validate_trainable_dataset.py \
  data/CoffeeFabric-Syn/heldout/P04_piecewise_mixed_occlusion_v7_heldout96
```

Both validators must report 84 train, 12 test, 96 ROI masks, and 6/6 left-right test views.

## 3. Freeze before training

The following command verifies every required artifact, rewrites the complete checksum
inventory, writes an anchored `HELDOUT_FREEZE.json` in each scene, and changes
`experiments/heldout_v34_matrix.json` from `preregistered` to `frozen` exactly once.

```bash
python tools/freeze_heldout_v34.py \
  --data_root data/CoffeeFabric-Syn/heldout
python tools/freeze_heldout_v34.py \
  --data_root data/CoffeeFabric-Syn/heldout --verify_only
```

Commit and push the generator, frozen datasets, locks and updated matrix before running a
model. Preserve that commit hash in the paper artifact. If large binary transfer is not
practical, create the freeze commit on the experiment server before training there.

## 4. Run the frozen held-out matrix on the server

After pulling the freeze commit, first verify the server copy and print the commands:

```bash
cd /home/featurize/work/3DGS_mac_new
conda activate megs
python tools/freeze_heldout_v34.py --verify_only
python tools/run_frozen_v34_matrix.py \
  --matrix experiments/heldout_v34_matrix.json
```

Then run one cell at a time. Example:

```bash
python tools/run_frozen_v34_matrix.py \
  --matrix experiments/heldout_v34_matrix.json \
  --scene P03 --method official_3dgs --seed 0 --execute

python tools/run_frozen_v34_matrix.py \
  --matrix experiments/heldout_v34_matrix.json \
  --scene P03 --method brgs_v34 --seed 0 --execute
```

Repeat methods for seeds 1 and 2, then repeat all six cells for P04. Do not use
`--allow_existing_train`; an interrupted incomplete directory must be renamed and retained
for audit before restarting cleanly.

## 5. Aggregate once, without tuning

```bash
python tools/aggregate_frozen_v34.py \
  --matrix experiments/heldout_v34_matrix.json \
  --output_dir comparisons/heldout_v34_final
cat comparisons/heldout_v34_final/summary.md
```

Report the complete result, including unfavorable metrics. P03/P04 may be called held-out
only if the freeze commit predates every model output timestamp and no parameter, scene,
seed, checkpoint or view subset is changed after results are seen.
