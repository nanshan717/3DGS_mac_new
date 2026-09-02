# CoffeeFabric-Syn scene generator

The P01 prototype uses only project-authored procedural geometry. It creates an actual
editable Blender scene, not a collection of fabricated metric values.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python tools/coffee_fabric_syn/generate_blender_scene.py -- \
  --output data/CoffeeFabric-Syn/prototype --seed 101
```

The generated `P01_flat_low_occlusion_v4_photoreal_attempt` directory contains the `.blend` scene, three rendered
paper-preview PNGs, the fabric ground-truth mesh, deterministic camera plans and train/test
transforms, metadata, empty modality directories reserved for the full exporter, and file
checksums. The first preview is a visual design gate; it is not yet the frozen benchmark.

Scientific labeling requirements:

- Call the scene synthetic and the vegetation coffee-inspired procedural proxy geometry.
- Do not claim that the plant model is botanically validated.
- Do not report the preview as a real field capture.
- Freeze scenes, seeds, versions and hashes before final comparative evaluation.

## Licensed external coffee model

After downloading and extracting an approved CC-licensed model, generate the asset-backed
scene with:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python tools/coffee_fabric_syn/generate_blender_scene.py -- \
  --output data/CoffeeFabric-Syn/prototype --seed 101 \
  --coffee-asset assets/coffee/<extracted-directory> --engine CYCLES
```

glTF/GLB normally uses `--asset-up-axis Z`. If inspection shows a sideways FBX/OBJ, pass
the source's declared `--asset-up-axis X` or `Y`; the generator never guesses from crown
dimensions because a mature coffee tree may legitimately be wider than it is tall.

The generator supports GLB, glTF, FBX and OBJ, normalizes the source to juvenile-scale
variants, uses linked mesh data, records the source hash, and writes `ASSET_ATTRIBUTION.json`.

To render a directly trainable RGB subset, add:

```text
--render-dataset --views 48 --dataset-resolution 800 --dataset-samples 48
```

Then validate every transform/image reference without installing training dependencies:

```bash
python tools/coffee_fabric_syn/validate_trainable_dataset.py \
  data/CoffeeFabric-Syn/prototype/P01_flat_low_occlusion_v6_paper_ready
```

The RGB renderer is the first trainability gate. Fabric masks, depth and normals remain a
separate export stage and must be produced from the same frozen `.blend` before benchmark
release.

## P01 V7 balanced training candidate

V7 retains the V6 geometry, materials, licensed coffee asset and seed, while replacing the
legacy sparse camera protocol with 96 paired views (84 train, 12 internal holdout). Test
views are balanced across the left and right trajectories, and a deterministic field-aware
`points3d.ply` replaces the generic `[-1.3, 1.3]` random initialization cube.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python tools/coffee_fabric_syn/generate_blender_scene.py -- \
  --output data/CoffeeFabric-Syn/prototype --seed 101 \
  --dataset-version v7 --views 96 \
  --coffee-asset assets/coffee/a_coffee_tree/a_coffee_tree.glb \
  --asset-license CC-BY-4.0 --engine CYCLES \
  --preview-resolution 800 --preview-samples 48 \
  --render-dataset --dataset-resolution 800 --dataset-samples 48
```

The validator enforces the 84/12 split, a 6/6 left-right test balance, non-degenerate camera
baselines, complete PNG references and the presence of the deterministic initialization
cloud. V6 is retained as a development record and must not be silently replaced by V7.

Render visible ground ROI masks from the frozen V7 scene with:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  data/CoffeeFabric-Syn/prototype/P01_flat_low_occlusion_v7_balanced96/P01_flat_low_occlusion_v7_balanced96.blend \
  --background --python tools/coffee_fabric_syn/render_ground_roi_masks.py -- \
  data/CoffeeFabric-Syn/prototype/P01_flat_low_occlusion_v7_balanced96 --resolution 800
```

These masks are synthetic ground-truth annotations: visible terrain and protective fabric are
white; coffee plants, weeds, root guards, posts and sky are black. BR-GS v3.1/v3.2 requires them
and applies the ROI to both surface fitting and floater suppression:

```text
--bsr_v31
```

For the geometry-safe preset used after the V7 15k diagnostic, use:

```bash
--bsr_v32
```

V3.2 keeps ROI-aware surface fitting, applies direct distance correction only beyond a
dead zone, and records reconstruction-only image-space gradients for densification. The
older `--bsr_v3` and `--bsr_v31` presets remain unchanged for ablation reproducibility.

V3.3 replaces direct point displacement with conservative opacity-only suppression:

```bash
--bsr_v33
```

Only visible, ROI-approved Gaussians beyond the detached surface-distance margin receive
the floater penalty. This branch cannot update Gaussian positions or the fitted surface,
and densification continues to use reconstruction gradients only.

V3.4 adds one auditable pruning event followed by reconstruction-only recovery:

```bash
--bsr_v34
```

At the pruning iteration, a Gaussian must simultaneously have low opacity, exceed the
normalized Bernstein-surface distance margin, and satisfy the ground-ROI consensus in
multiple training views. A hard maximum removal fraction is enforced. The resolved
thresholds and before/after counts are written to `bsr_v34_pruning.json`; densification
and BR-GS regularization are disabled after pruning.

The experimental `--bsr_v35` preset retains this pruning stage and adds bounded,
photometrically guarded weak surface recovery. See
`docs/BRGS_V35_DEVELOPMENT_PROTOCOL.md` for its safeguards and split protocol.

## P02 frozen validation candidate

P02 is a new scene design, not P01 with a renamed random seed. It uses seed 202, a
continuous undulating surface, mild directional slope, 15 independently placed and scaled
plants, and medium weed occlusion. P01 remains the development scene; do not change the
frozen BR-GS v3.4 preset after observing P02.

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python tools/coffee_fabric_syn/generate_blender_scene.py -- \
  --output data/CoffeeFabric-Syn/prototype --seed 202 \
  --scene-profile p02 --dataset-version v7 --views 96 \
  --coffee-asset assets/coffee/a_coffee_tree/a_coffee_tree.glb \
  --asset-license CC-BY-4.0 --engine CYCLES \
  --preview-resolution 800 --preview-samples 48 \
  --render-dataset --dataset-resolution 800 --dataset-samples 48
```

Render its ROI masks and validate the trainable structure exactly as for P01. P02 is a
validation candidate, not the final hidden test set.

## P03/P04 pre-registered held-out scenes

P03 and P04 are the first scene-level held-out evaluation pair. Their specifications live
in `tools/coffee_fabric_syn/heldout_profiles.py`: P03 uses seed 303, cross-slope terrain,
transverse wrinkles and high occlusion; P04 uses seed 404, piecewise crossfall, a local
uplift/edge sag and mixed occlusion. These values must not be changed after any P03/P04
reconstruction is trained or evaluated.

Generate both complete RGB datasets with the commands in
`docs/HELDOUT_V34_EXPERIMENT_PROTOCOL.md`. Render the ROI masks from the generated `.blend`
files, validate both structures, then run `tools/freeze_heldout_v34.py`. The held-out matrix
is deliberately marked `preregistered` until this freeze succeeds, and the matrix runner
will refuse to train it in that state.
