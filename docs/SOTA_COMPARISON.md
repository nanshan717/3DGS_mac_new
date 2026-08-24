# Surface-reconstruction comparison set

Primary comparison:

1. Official 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
2. 2D Gaussian Splatting: https://github.com/hbb1/2d-gaussian-splatting
3. PGSR: https://github.com/zju3dv/PGSR
4. Gaussian Opacity Fields: https://github.com/autonomousvision/gaussian-opacity-fields
5. BR-GS v3 (this repository)

Extended comparison, subject to compute and dependency availability:

6. SuGaR: https://github.com/Anttwo/SuGaR
7. Sorted Opacity Fields: https://github.com/r4dl/SOF

Use the same source images, scene-level split, COLMAP reconstruction where accepted,
resolution, and exposure preprocessing. Record native training time and peak memory; do
not force identical iteration counts when methods define materially different stages.
Instead report both each official recommended configuration and a matched wall-clock
budget where practical.

Convert native mesh/surfel/Gaussian outputs into a common sampled point-cloud protocol for
accuracy, completeness, Chamfer distance, F-score, and normal consistency. Also report
each method's native mesh result. Rendering metrics use the same held-out cameras. Fabric
metrics use the ground-truth fabric mask/mesh only and must not include vegetation.

