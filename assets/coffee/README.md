# Licensed coffee asset staging directory

Download the original archive for the selected model through the authenticated source site,
extract it under this directory, and retain the original license/readme files. Do not rename
or discard provenance files.

Current candidate:

- Title: `A coffee tree`
- Author: `rvezy`
- Source: https://sketchfab.com/3d-models/a-coffee-tree-045dba854c8d4b9e8a5dff2d18892df1
- Displayed license at review time: `CC BY`

The asset itself is intentionally not committed by this implementation. Its license must be
confirmed again at download time. After extraction, invoke the generator with
`--coffee-asset assets/coffee/<extracted-directory>`.

This candidate includes a `Cobblestone` display base that is not part of the plant. The
generator excludes that material by default and records the exclusion in its attribution
manifest. Use `--asset-exclude-materials ''` only after visually confirming a different asset.
