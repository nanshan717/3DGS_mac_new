#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/featurize/work/3DGS_mac_new
PY=/environment/miniconda3/envs/megs/bin/python
OUT_ROOT="$ROOT/output_ablation/brgs_v34_noprune_15k"
RESULT_ROOT="$ROOT/baseline_results/brgs_v34_noprune_15k"
STOP_FILE="$ROOT/STOP_AFTER_CURRENT_ABLATION"

declare -A DATASETS=(
  [P01]="/home/featurize/work/3DGS_mac_new/data/CoffeeFabric-Syn/prototype/P01_flat_low_occlusion_v7_balanced96"
  [P02]="/home/featurize/work/3DGS_mac_new/data/CoffeeFabric-Syn/prototype/P02_undulating_medium_occlusion_v7_balanced96"
  [P03]="/home/featurize/work/3DGS_mac_new/data/CoffeeFabric-Syn/heldout/P03_sloped_high_occlusion_v7_heldout96"
  [P04]="/home/featurize/work/3DGS_mac_new/data/CoffeeFabric-Syn/heldout/P04_piecewise_mixed_occlusion_v7_heldout96"
)

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$RESULT_ROOT"

for scene in P01 P02 P03 P04; do
  source_path=${DATASETS[$scene]}
  for seed in 0 1 2; do
    tag="${scene}_brgs_v34_noprune_15k_s${seed}"
    model="$OUT_ROOT/$tag"
    result="$RESULT_ROOT/${scene}_s${seed}"

    if [[ -s "$result/COMPLETE" ]]; then
      echo "Skipping completed cell: $scene seed=$seed"
      continue
    fi
    if [[ -e "$model" ]]; then
      echo "ERROR: incomplete output already exists: $model" >&2
      echo "Inspect it before deleting it and restarting this cell." >&2
      exit 2
    fi

    avail_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
    if (( avail_kb < 1300000 )); then
      echo "ERROR: less than 1.3 GB free; stopping before $scene seed=$seed" >&2
      exit 3
    fi

    mkdir -p "$result"
    {
      echo "ablation=BR-GS v3.4 without physical pruning"
      echo "single_change=--bsr_prune_max_fraction 0.0"
      echo "scene=$scene"
      echo "seed=$seed"
      echo "source_path=$source_path"
      echo "python=$PY"
      echo "git_commit=$(git rev-parse HEAD)"
      echo "train_py_sha256=$(sha256sum train.py | awk '{print $1}')"
      echo "started_utc=$(date -u +%FT%TZ)"
    } > "$result/protocol.txt"

    echo "========== $scene seed=$seed: train =========="
    "$PY" train.py \
      -s "$source_path" \
      -m "$model" \
      -r 1 \
      --eval \
      --iterations 15000 \
      --seed "$seed" \
      --deterministic \
      --bsr_v34 \
      --bsr_prune_max_fraction 0.0

    ply="$model/point_cloud/iteration_15000/point_cloud.ply"
    audit="$model/bsr_v34_pruning.json"
    [[ -s "$ply" && -s "$audit" && -s "$model/experiment_manifest.json" ]]

    "$PY" - "$audit" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["max_fraction"] == 0.0, audit
assert audit["points_removed"] == 0, audit
assert audit["points_before"] == audit["points_after"], audit
print("No-pruning audit: OK", audit["points_before"], "Gaussians")
PY

    echo "========== $scene seed=$seed: render test =========="
    "$PY" render.py -m "$model" --iteration 15000 --skip_train

    echo "========== $scene seed=$seed: image metrics =========="
    "$PY" /home/featurize/work/3DGS_mac_new/metrics.py -m "$model"

    echo "========== $scene seed=$seed: geometry metrics =========="
    "$PY" /home/featurize/work/3DGS_mac_new/eval_geometry_gt.py \
      -s "$source_path" \
      -m "$model" \
      --iteration 15000 \
      --save_json

    [[ -s "$model/results.json" ]]
    [[ -s "$model/per_view.json" ]]
    [[ -s "$model/gt_geometry_results_iter-15000.json" ]]

    cp -f "$model/results.json" "$result/"
    cp -f "$model/per_view.json" "$result/"
    cp -f "$model/gt_geometry_results_iter-15000.json" "$result/"
    cp -f "$model/experiment_manifest.json" "$result/"
    cp -f "$model/bsr_v34_pruning.json" "$result/"
    cp -f "$model/cfg_args" "$result/"
    date -u +completed_utc=%FT%TZ >> "$result/protocol.txt"

    # Image metrics are already recorded; these PNGs are reproducible from the PLY.
    rm -rf -- "$model/test"
    gzip -1 -- "$ply"
    echo "complete" > "$result/COMPLETE"
    sync
    du -sh "$model"
    df -h "$ROOT"

    if [[ -e "$STOP_FILE" ]]; then
      rm -f -- "$STOP_FILE"
      echo "Safe-stop requested; stopped after completing $scene seed=$seed."
      exit 0
    fi
  done
done

echo "All BR-GS v3.4 no-pruning ablation cells completed."
