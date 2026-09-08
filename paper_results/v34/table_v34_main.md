# BR-GS v3.4 main quantitative table

Mean ± sample standard deviation over three paired training seeds. Development and
pre-registered held-out scenes are reported separately.

| Split | Scene | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Points (k)↓ | Chamfer (m)↓ | Completeness (m)↓ | F@5cm↑ |
|---|---|---|---|---|---|---|---|---|---|
| Development | P01 | Official 3DGS | 18.889 ± 0.631 | 0.7946 ± 0.0011 | 0.3762 ± 0.0023 | 676 ± 43 | **0.0660 ± 0.0003** | 0.0396 ± 0.0006 | **0.4399 ± 0.0054** |
| Development | P01 | BR-GS v3.4 | **19.128 ± 0.435** | **0.7955 ± 0.0011** | **0.3724 ± 0.0050** | **485 ± 26** | 0.0664 ± 0.0002 | **0.0379 ± 0.0001** | 0.4353 ± 0.0017 |
| Development | P02 | Official 3DGS | 17.897 ± 0.160 | **0.7078 ± 0.0004** | 0.3987 ± 0.0010 | 1175 ± 156 | 0.0656 ± 0.0009 | 0.0364 ± 0.0002 | 0.4510 ± 0.0021 |
| Development | P02 | BR-GS v3.4 | **18.084 ± 0.125** | 0.7074 ± 0.0013 | **0.3946 ± 0.0016** | **815 ± 112** | **0.0641 ± 0.0009** | **0.0348 ± 0.0004** | **0.4535 ± 0.0044** |
| Held-out | P03 | Official 3DGS | 21.614 ± 0.303 | 0.8874 ± 0.0012 | 0.1696 ± 0.0024 | 1338 ± 69 | **0.0367 ± 0.0002** | 0.0281 ± 0.0001 | **0.7524 ± 0.0009** |
| Held-out | P03 | BR-GS v3.4 | **21.628 ± 0.039** | **0.8887 ± 0.0011** | **0.1675 ± 0.0026** | **1271 ± 9** | 0.0383 ± 0.0003 | **0.0266 ± 0.0003** | 0.7441 ± 0.0032 |
| Held-out | P04 | Official 3DGS | **23.610 ± 1.366** | **0.8971 ± 0.0068** | **0.2020 ± 0.0067** | 830 ± 20 | 0.0453 ± 0.0046 | 0.0240 ± 0.0004 | **0.7613 ± 0.0102** |
| Held-out | P04 | BR-GS v3.4 | 23.457 ± 1.234 | 0.8894 ± 0.0116 | 0.2098 ± 0.0165 | **788 ± 21** | **0.0415 ± 0.0041** | **0.0233 ± 0.0001** | 0.7487 ± 0.0072 |

Bold marks the better mean within each scene and metric; it does not denote
statistical significance. Lower is better for LPIPS, points, Chamfer, and completeness.
