# CoffeeFabric-Syn specification

CoffeeFabric-Syn is a scene-level benchmark for reconstructing protective agricultural
fabric beneath occluding coffee plants. Synthetic data supplements, but does not replace,
a small real capture set.

Create a deterministic fabric layout and camera plan with:

```bash
python tools/coffee_fabric_syn/generate_layout.py \
  --output data/CoffeeFabric-Syn/dev/wrinkle_0007 \
  --seed 7 --morphology wrinkle --views 200
```

This produces metric fabric ground truth but deliberately marks the scene `layout_only`.
It must be imported into Blender/BlenderProc, populated with licensed coffee/soil assets,
and rendered before it is a trainable dataset.

## Initial release target

- 24 independently generated scenes: 12 development, 4 validation, 8 held-out test.
- 150–300 overlapping mobile-camera views per scene.
- Three fabric materials: nonwoven weed mat, plastic mulch, and shade net.
- Six morphology groups: flat, slope, sag, uplift, wrinkle, and damaged/buried edge.
- Low, medium, and high vegetation occlusion.
- Stable, hard-shadow, overcast, and exposure-varying illumination.
- Flat, sloped, undulating, and piecewise terrain.

Splits are by complete 3-D scene and random seed, never by frames from the same scene.

## Required scene layout

```text
scene_name/
  images/
  bsr_masks/             # binary fabric masks, same stems as images
  depth/                 # full-scene metric depth
  fabric_depth/          # fabric-only metric depth
  normals/
  sparse/0/              # COLMAP model used by 3DGS
  ground_truth/
    fabric_mesh.ply
    fabric_points.ply       # dense uniformly sampled evaluation points
    scene_mesh.ply
  transforms_train.json
  metadata.json
```

`metadata.json` records units, generator commit, seed, material, deformation parameters,
occlusion ratio, lighting, camera trajectory, and train/test frame lists. Camera poses,
fabric mesh, masks, depth, and normals must originate from the same frozen scene state.

## Capture design

Use a mobile-height path through planting rows, not only a top-down orbit. Include oblique
views that see beneath foliage, keep at least 70 percent adjacent-frame overlap, lock
geometry during each sequence, and render separate train/test trajectories. BlenderProc
is recommended because it exports RGB, depth, normals, segmentation, and poses from the
same scene. Only assets with redistribution-compatible licenses may be included in a
released dataset; otherwise publish generation scripts and asset source instructions.

## Real controlled subset

Capture coffee seedlings above a known protective mat in flat, sag, uplift, wrinkle, and
partial-burial conditions. Include a metric scale or AprilTag. Acquire an unobstructed
reference scan/photogrammetry model first, then add plant occlusion without changing the
fabric. This supports metric geometry evaluation and a defensible synthetic-to-real study.
