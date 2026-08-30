# CoffeeFabric-Syn (development dataset card)

CoffeeFabric-Syn is a synthetic benchmark under development for evaluating protective-
fabric reconstruction beneath coffee-inspired vegetation. It must not be represented as a
real coffee plantation capture. P01 is the development/tuning scene; P02 is the first
frozen validation candidate and is not a hidden final-test scene.

## Provenance

Terrain, fabric, weeds, guards, lighting and layouts are created procedurally by the
accompanying Blender Python script. V5 and later scenes instance the separately licensed
coffee-tree asset recorded in each scene's `ASSET_ATTRIBUTION.json`; its source license and
hash must remain with every release. Project-owned field photographs informed qualitative
layout and morphology only: no photograph pixels or textures are copied into the scenes.
Fabric deformation is generated independently of the Bernstein representation used by
BR-GS. Exact random seeds, Blender version, scene metadata and SHA-256 checksums accompany
each generated scene.

## Intended use

P01 is for visual review, pipeline validation and method development. P02 uses seed 202,
undulating terrain, medium occlusion and an independent 15-plant layout; it may be used for
validation without changing the frozen BR-GS v3.4 preset. A release still requires new
unseen scene-level test scenes, failure cases and method-independent metrics. Synthetic
results must be reported separately from controlled real and field data.

## Licensing status

Dataset and generator licensing is intentionally marked as pending project/institutional
review. Do not publish the dataset under an assumed license until that review is complete.
