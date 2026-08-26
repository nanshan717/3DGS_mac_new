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
