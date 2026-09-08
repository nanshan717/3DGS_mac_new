# BR-GS v3.4 results, discussion, and limitations draft

## Evaluation protocol and reporting boundary

We compare the frozen BR-GS v3.4 configuration with the official 3DGS baseline at
15,000 iterations. Each method is trained with seeds 0, 1, and 2, and all values are
reported as the arithmetic mean plus or minus the sample standard deviation across those
three paired seeds. P01 and P02 were used during method development and are therefore
reported only as development evidence. P03 and P04 were pre-registered, rendered,
structurally validated, and checksum-frozen before any model was trained; these scenes
constitute the held-out evaluation. No BR-GS hyperparameter, scene definition, camera
split, checkpoint, or seed was changed after held-out results were observed.

## Development results

On P01, BR-GS reduced the mean Gaussian count from 676k to 485k (28.23%) while increasing
PSNR by 0.239 dB and reducing LPIPS by 0.0038. Completeness improved from 0.0397 m to
0.0379 m, although Chamfer L1 increased by 0.0004 m and F@5cm decreased by 0.0046.

On P02, BR-GS reduced the mean Gaussian count from 1.175M to 0.815M (30.67%), increased
PSNR by 0.187 dB, and reduced LPIPS by 0.0041. Completeness and Chamfer L1 both improved,
while F@5cm increased by 0.0025. Across these two development scenes, the scene-balanced
descriptive averages correspond to a 29.45% point-count reduction, a 0.213 dB PSNR gain,
and a 0.0039 LPIPS reduction. These development values are not presented as unbiased
generalization estimates.

## Pre-registered held-out results

On P03, BR-GS reduced the mean Gaussian count from 1.338M to 1.271M (5.02%). Mean PSNR was
effectively unchanged (+0.013 dB), SSIM increased by 0.0014, and LPIPS decreased by
0.0020. Completeness improved by 0.0015 m. In contrast, Chamfer L1 increased by 0.0016 m
and F@5cm decreased from 0.7524 to 0.7441.

On P04, BR-GS reduced the mean Gaussian count from 830k to 788k (5.16%). Completeness and
Chamfer L1 improved by 0.0007 m and 0.0039 m, respectively. Mean PSNR decreased by
0.153 dB, SSIM decreased by 0.0076, LPIPS increased by 0.0078, and F@5cm decreased from
0.7613 to 0.7487. Image-space performance on P04 was seed-sensitive: paired PSNR changes
for seeds 0, 1, and 2 were +1.022, -1.419, and -0.063 dB, respectively.

Across the two held-out scenes, a scene-balanced descriptive average gives a 5.09%
point-count reduction, a -0.070 dB PSNR change, a +0.0029 LPIPS change, and a -0.0011 m
Chamfer change. With only two held-out scenes, these aggregate values are descriptive and
are not treated as evidence of statistical significance.

## Qualitative protocol

Qualitative comparisons use the same held-out camera indices for ground truth, official
3DGS, and BR-GS. The main panel fixes test view 00005 and the primary seed s0, with an
identical center crop for both methods. A supplementary overview reports the first,
middle, and last test indices (00000, 00005, and 00011), and a second supplementary panel
shows all three registered seeds for view 00005. These choices are index-based rather
than selected from per-view metric rankings. The qualitative examples are illustrative;
the quantitative conclusions continue to use all 12 test views and all three seeds.

## Discussion

The most consistent effects of BR-GS v3.4 are reduced model size and lower completeness
error: point count and completeness improved on all four scenes. The remaining effects
are scene-dependent. Image quality improved on both development scenes and was mixed on
held-out scenes. Chamfer improved on P02 and P04 but degraded on P01 and P03. F@5cm
improved only on P02. Thus, the results support describing BR-GS as a compactness-oriented
regularizer that can preserve image quality and improve coverage, but they do not support
a claim of universal image-quality or geometry improvement.

The gap between the approximately 29% development-set compression and 5% held-out
compression also indicates that the final model-size effect depends on the scene and its
training trajectory. We therefore report per-scene point counts rather than presenting a
single compression factor as an intrinsic property of the method.

## Limitations

The evaluation contains two development scenes and only two pre-registered held-out
scenes, all from the synthetic CoffeeFabric-Syn setting. Three training seeds expose
substantial stochastic variation, particularly on P04, but are insufficient for strong
distributional or significance claims. Completeness improves consistently, whereas
accuracy, Chamfer, and thresholded F-scores do not move together; this reflects different
failure sensitivities and requires reporting the full metric set. The current evidence
also does not establish performance on real coffee farms, other crop geometries, or
substantially different capture conditions. Qualitative comparisons must use identical
held-out cameras and must be selected without suppressing unfavorable examples.

## Claim-safe summary

Frozen BR-GS v3.4 consistently reduced Gaussian count and completeness error across four
synthetic fabric-support scenes. On the pre-registered held-out pair it reduced mean point
count by approximately 5% while the scene-balanced PSNR change was -0.07 dB. Other image
and geometry metrics were mixed across scenes, so all per-scene results and unfavorable
metrics are reported.
