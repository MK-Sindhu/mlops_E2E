#!/usr/bin/env bash
# Run a small experiment sweep that exercises both model knobs (max_depth,
# learning_rate, n_estimators, quantization) AND a data knob (test_size).
# Each sweep modifies configs/config.yaml temporarily, runs `dvc repro`, and
# saves the result as a `dvc exp save` snapshot. The original config is
# always restored on exit.
#
# Why this script exists: `dvc exp run -S key=value` is broken in DVC 3.42
# in this environment (omegaconf / Repo.stage AttributeError). This script
# achieves the same outcome — populated `dvc exp show` — without depending
# on that code path.
#
# Usage:
#   bash scripts/run_dvc_experiments.sh
set -euo pipefail

CONFIG="configs/config.yaml"
BACKUP="$(mktemp -t config.yaml.backup.XXXXXX)"
cp "$CONFIG" "$BACKUP"

# Always restore the original config when this script exits (success or fail).
trap 'echo "Restoring $CONFIG from backup."; cp "$BACKUP" "$CONFIG"; rm -f "$BACKUP"' EXIT

run_experiment () {
  local name="$1" ; shift
  local description="$1" ; shift
  echo
  echo "=============================================================="
  echo "EXPERIMENT: $name"
  echo "  $description"
  echo "=============================================================="

  # Build the edit script in a temp file so multi-line python is reliable.
  local pyfile
  pyfile=$(mktemp -t dvc_exp_edit.XXXXXX.py)
  {
    echo "import yaml"
    echo "with open('$CONFIG') as f:"
    echo "    cfg = yaml.safe_load(f)"
    for stmt in "$@"; do
      echo "$stmt"
    done
    echo "with open('$CONFIG', 'w') as f:"
    echo "    yaml.safe_dump(cfg, f, sort_keys=False)"
    echo "print('Wrote new $CONFIG')"
  } > "$pyfile"

  python "$pyfile"
  rm -f "$pyfile"

  dvc repro train evaluate
  dvc exp save -n "$name" -m "$description" -f
}

# 1) Shallower trees, more of them — slower training, often smoother boundary.
run_experiment "exp_shallow_more_trees" \
  "max_depth=4, n_estimators=200 (smaller individual trees, more of them)" \
  'cfg["model"]["params"]["max_depth"] = 4' \
  'cfg["model"]["params"]["n_estimators"] = 200'

# 2) Lower learning rate, deeper trees — different bias/variance trade-off.
run_experiment "exp_deep_slow_lr" \
  "max_depth=10, learning_rate=0.05 (deeper trees, slower learning rate)" \
  'cfg["model"]["params"]["max_depth"] = 10' \
  'cfg["model"]["params"]["learning_rate"] = 0.05'

# 3) Quantization OFF — XGBoost defaults; tests that the quantize toggle
#    actually matters. Touches model.optimization.quantize.
run_experiment "exp_no_quantization" \
  "quantize=false (XGBoost defaults: tree_method=hist with max_bin=256)" \
  'cfg["model"]["optimization"]["quantize"] = False'

# 4) Different data version: 30/70 train/test split (vs. default 80/20).
#    Tests pipeline reproducibility under a different data partition.
run_experiment "exp_data_split_30pct" \
  "test_size=0.3 (data version: 70% train / 30% test split)" \
  'cfg["data"]["test_size"] = 0.3'

echo
echo "All experiments complete. Original $CONFIG restored."
echo "View results with:  dvc exp show"
