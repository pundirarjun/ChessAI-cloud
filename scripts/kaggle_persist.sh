#!/usr/bin/env bash
set -euo pipefail

# Persist the clean AlphaZero run state (manifest, checkpoints, replay,
# run_state) to a Kaggle Dataset so training progress survives session
# death.  The Kaggle working directory is wiped when a session stops, so
# every saved iteration is uploaded here and re-downloaded on the next
# session.
#
# Usage: kaggle_persist.sh restore|save [project_root]
#
# Enabled automatically inside Kaggle notebooks (detected via /kaggle/input
# or KAGGLE_KERNEL_RUN_TYPE) when KAGGLE_USERNAME and KAGGLE_KEY are set.
#   KAGGLE_DATASET_SLUG  override the default "$KAGGLE_USERNAME/chessai-clean-az-run"
#   PERSIST=0            disable entirely

ACTION="${1:-}"
PROJECT_ROOT="${2:-/kaggle/working/ChessAI-cloud}"
case "$ACTION" in
  restore|save) ;;
  *) echo "usage: $0 restore|save [project_root]" >&2; exit 2 ;;
esac
cd "$PROJECT_ROOT"

if [ "${PERSIST:-1}" = "0" ]; then
  echo "[persist] disabled via PERSIST=0"
  exit 0
fi
if ! { [ -d /kaggle/input ] || [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]; }; then
  echo "[persist] not a Kaggle notebook; skipping $ACTION"
  exit 0
fi
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  echo "[persist] KAGGLE_USERNAME/KAGGLE_KEY missing; skipping $ACTION"
  exit 0
fi

SLUG="${KAGGLE_DATASET_SLUG:-$KAGGLE_USERNAME/chessai-clean-az-run}"
command -v kaggle >/dev/null 2>&1 || python -m pip install --quiet kaggle

RUN_DIR="$(python -c "import sys; from az.config import RunConfig; print(RunConfig.load_json(sys.argv[1]).root.as_posix())" "$PROJECT_ROOT/configs/clean_az.json")"

free_kb() { df -k "$PROJECT_ROOT" | awk 'NR==2{print $4}'; }

restore_run() {
  if [ -f "$RUN_DIR/run_state.json" ]; then
    echo "[persist] local run state already present; keeping it (dataset untouched)"
    return 0
  fi
  # Fresh session: 19 GB total. Download + unzip needs roughly 2x the run size.
  local kb
  kb="$(free_kb)"
  if [ "$kb" -lt $((6 * 1024 * 1024)) ]; then
    echo "[persist] only $((kb / 1024 / 1024)) GB free (<6 GB); skipping restore"
    return 0
  fi
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  if ! kaggle datasets download "$SLUG" -p "$tmp" >/dev/null 2>&1; then
    echo "[persist] no dataset '$SLUG' yet (first run?); skipping restore"
    return 0
  fi
  mkdir -p "$tmp/run"
  unzip -q -o "$tmp"/*.zip -d "$tmp/run"
  if [ ! -f "$tmp/run/manifest.json" ] || [ ! -f "$tmp/run/run_state.json" ]; then
    echo "[persist] dataset '$SLUG' has no recognizable run state; skipping restore"
    return 0
  fi
  # Compare the stored run's config hash with the current config so a
  # config change never resurrects an incompatible run state silently.
  local stored current
  stored="$(sed -n 's/.*"config_sha256"[[:space:]]*:[[:space:]]*"\([0-9a-fA-F]*\)".*/\1/p' "$tmp/run/manifest.json" | head -1)"
  current="$(python -c "import hashlib, sys; from az.config import RunConfig; print(hashlib.sha256(RunConfig.load_json(sys.argv[1]).canonical_json().encode('utf-8')).hexdigest())" "$PROJECT_ROOT/configs/clean_az.json")"
  if [ -z "$stored" ] || [ "$stored" != "$current" ]; then
    echo "[persist] dataset state was created with a different config; skipping restore."
    echo "[persist] for a fresh start delete the dataset version: https://www.kaggle.com/datasets/$SLUG"
    return 0
  fi
  mkdir -p "$RUN_DIR"
  cp -a "$tmp/run/." "$RUN_DIR/"
  local iter
  iter="$(python -c "import json; print(json.load(open('$RUN_DIR/run_state.json'))['iteration'])")"
  echo "[persist] restored run state (iteration $iter) from $SLUG"
}

save_run() {
  if [ ! -f "$RUN_DIR/run_state.json" ]; then
    echo "[persist] no run state to save yet"
    return 0
  fi
  local iter run_bytes need_kb kb
  iter="$(python -c "import json; print(json.load(open('$RUN_DIR/run_state.json'))['iteration'])")"
  run_bytes="$(du -sb "$RUN_DIR" | awk '{print $1}')"
  need_kb=$((run_bytes / 1024 + 1048576)) # run size + 1 GB headroom for the zip
  kb="$(free_kb)"
  if [ "$kb" -lt "$need_kb" ]; then
    echo "[persist] WARN: $((kb / 1024 / 1024)) GB free, need ~$((need_kb / 1024 / 1024)) GB to package; skipping save"
    return 0
  fi
  echo "[persist] packaging run state: $((run_bytes / 1024 / 1024)) MB, $((kb / 1024 / 1024)) GB free"
  printf '{"title":"ChessAI clean AZ run state","id":"%s","licenses":[{"name":"CC BY-SA 4.0"}]}\n' \
    "$SLUG" > "$RUN_DIR/dataset-metadata.json"
  if kaggle datasets metadata "$SLUG" >/dev/null 2>&1; then
    kaggle datasets version -p "$RUN_DIR" -m "iteration $iter auto-save"
  else
    kaggle datasets create -p "$RUN_DIR" -m "iteration $iter auto-save" --public
  fi
  echo "[persist] saved run state (iteration $iter) to https://www.kaggle.com/datasets/$SLUG"
}

case "$ACTION" in
  restore) restore_run ;;
  save) save_run ;;
esac
