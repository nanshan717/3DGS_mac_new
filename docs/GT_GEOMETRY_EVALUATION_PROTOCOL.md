# CoffeeFabric-Syn ground-truth geometry protocol

This protocol is frozen before reporting oracle geometry results. It applies only to
synthetic scenes with a metric `ground_truth/fabric_mesh.ply`; it must not be used for
real captures without a registered reference scan.

The evaluator samples 250,000 points from mesh triangles in proportion to surface area
with seed 3401. Reconstructed Gaussian centres must have opacity at least 0.05, lie in
the GT horizontal support (2 cm tolerance), and lie no more than 15 cm above the local
fabric height. The upper bound excludes coffee canopy centres; points arbitrarily far
below the fabric remain included and are penalised. Candidate centres are averaged in a
fixed 5 mm voxel grid before evaluation so a method cannot improve a score merely by
placing more duplicate Gaussians in the same region.

Report both directions:

- Accuracy: reconstructed fabric candidate to sampled GT mesh distance.
- Completeness: sampled GT mesh to reconstructed fabric candidate distance.
- symmetric Chamfer-L1 and Chamfer-L2;
- precision, recall, and F-score at fixed 1 cm, 2 cm, and 5 cm thresholds.

Use identical sampled GT points and filtering parameters for all compared methods. The
implementation is point-to-densely-sampled-mesh; disclose this wording rather than
calling it exact point-to-triangle distance.

```bash
python eval_geometry_gt.py \
  -s data/CoffeeFabric-Syn/prototype/P02_undulating_medium_occlusion_v7_balanced96 \
  -m /path/to/official/output/P02_official_3dgs_15k output/P02_brgs_v34_15k_s0 \
  --iteration 15000 --save_json
```
