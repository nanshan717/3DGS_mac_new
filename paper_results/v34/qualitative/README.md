# BR-GS v3.4 qualitative figures

The figures use only the pre-registered held-out scenes P03 and P04.

Selection protocol:

- Main figure: fixed test view `00005`, fixed primary seed `s0`, and a fixed center
  crop `(x=180, y=180, width=440, height=440)` in the 800 x 800 source image.
- Fixed-view overview: `00000`, `00005`, and `00011` (first, middle, and last
  test indices), all using seed `s0`.
- Multi-seed panel: fixed test view `00005`, showing seeds `s0`, `s1`, and `s2`.

The selections are index-based, not chosen from per-view metric rankings. The main paper
must retain the all-view, three-seed quantitative table; these images are illustrative.
`INPUT_SHA256SUMS` fixes all 180 source PNG/JSON files used by the generator.

Suggested main-figure caption:

> Qualitative comparison on the pre-registered held-out scenes P03 and P04. We show the
> fixed test view 00005 for the primary seed s0, with identical center crops. BR-GS v3.4
> uses fewer Gaussians, while image-space differences relative to official 3DGS are
> subtle and scene-dependent. Quantitative results average all 12 test views and three
> paired training seeds.

Suggested supplementary caption:

> Multi-seed qualitative comparison for fixed held-out test view 00005. All three
> pre-registered seeds are shown to expose stochastic variation and avoid selecting a
> favorable individual run.
