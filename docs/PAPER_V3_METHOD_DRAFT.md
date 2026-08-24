# BR-GS v3 method draft

This file is a method-writing baseline. Numerical claims must not be copied into the paper
until the frozen experiments have completed.

Let `n` be an automatically estimated support normal and let `c` be the support-frame
origin. The signed support height of Gaussian center `mu_k` is

```text
h_k = n^T (mu_k - c).
```

Candidate confidence combines a soft height weight, opacity confidence, optional local
density confidence, and optional multi-view fabric ROI confidence:

```text
w_k = w_height,k w_opacity,k w_density,k w_roi,k.
```

The support region is represented by a grid of tensor-product Bernstein patches. Control
points retain fixed in-plane coordinates and move only along `n`, preventing tangential
drift. Adjacent patches share a positional seam penalty; second differences of each
control net provide a low-frequency fabric prior.

Distances are normalized by the robust support-region diagonal. A robust point-to-surface
loss attracts confident fabric Gaussians to the support, while a surface-to-point term
prevents patch collapse and poor coverage. A separate opacity loss suppresses distant,
low-confidence floaters rather than excluding them and claiming they were geometrically
corrected.

```text
L = L_rgb
  + lambda_g(t) [L_point_to_surface + lambda_cov L_surface_to_point]
  + lambda_ctrl L_control
  + lambda_seam L_continuity
  + lambda_float(t) L_floater.
```

Training is staged: photometric warm-up, surface initialization, gradual geometric
activation during late densification, full geometric refinement after densification, and
a reduced-weight final photometric refinement. Synthetic ground-truth masks are used only
in the annotated ROI setting; the annotation-free setting uses geometric confidence and
must be reported separately.

The learned training surface is not used as the sole evaluator. Geometry evaluation uses
held-out ground truth where available and an independently post-fitted surface plus local
PCA diagnostics where ground truth is unavailable.

