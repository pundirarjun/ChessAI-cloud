#!/usr/bin/env bash
set -euo pipefail

# Persist the clean AlphaZero run state (manifest, checkpoints, replay,
# run_state) to a Kaggle Dataset so training progress survives session
# death.  The Kaggle working directory is wiped when a session stops, so
# every saved iteration is uploaded here and re-downloaded on the next
# session.
#
# Upload layout: the run dir is packed into a single run_state.zip (the
# kaggle CLI's default dir-mode skips subdirectories, and per-directory
# archives have inconsistent layouts), uploaded next to the required
# dataset-metadata.json.  Restore reverses that exact layout.
#
# Usage: kaggle_persist.sh restore|save [project_root]
#
# Enabled automatically inside Kaggle notebooks (detected via /kaggle/input
# or KAGGLE_KERNEL_RUN_TYPE); credentials come from KAGGLE_USERNAME/KAGGLE_KEY
# or, if unset, from ~/.kaggle/kaggle.json.
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
# Credentials: prefer env vars; fall back to ~/.kaggle/kaggle.json, which
# Kaggle notebooks mount even when KAGGLE_USERNAME/KAGGLE_KEY are unset.
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    KAGGLE_USERNAME="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["username"])' "$HOME/.kaggle/kaggle.json")"
    KAGGLE_KEY="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"])' "$HOME/.kaggle/kaggle.json")"
    export KAGGLE_USERNAME KAGGLE_KEY
    echo "[persist] credentials loaded from $HOME/.kaggle/kaggle.json"
  else
    echo "[persist] no credentials (env vars or $HOME/.kaggle/kaggle.json); skipping $ACTION"
    exit 0
  fi
fi

SLUG="${KAGGLE_DATASET_SLUG:-$KAGGLE_USERNAME/chessai-clean-az-run}"
command -v kaggle >/dev/null 2>&1 || python -m pip install --quiet kaggle

RUN_DIR="$(python -c "import sys; from az.config import RunConfig; print(RunConfig.load_json(sys.argv[1]).root.as_posix())" "$PROJECT_ROOT/configs/clean_az.json")"

CLEANUP_DIRS=()
cleanup() {
  local d
  for d in "${CLEANUP_DIRS[@]}"; do rm -rf "$d"; done
}
trap cleanup EXIT

# Translate a path for Windows Python when running under Git bash (local
# tests); a no-op on Kaggle/Linux where cygpath does not exist.
wpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

free_kb() { df -k "$PROJECT_ROOT" | awk 'NR==2{print $4}'; }

restore_run() {
  if [ -f "$RUN_DIR/run_state.json" ]; then
    echo "[persist] local run state already present; keeping it (dataset untouched)"
    return 0
  fi
  # Session start has ~19 GB free; the download plus unpacked run dir must
  # both fit (checked precisely after download, before unpacking).
  local kb
  kb="$(free_kb)"
  if [ "$kb" -lt $((12 * 1024 * 1024)) ]; then
    echo "[persist] only $((kb / 1024 / 1024)) GB free (<12 GB); skipping restore"
    return 0
  fi
  local tmp
  tmp="$(mktemp -d)"
  CLEANUP_DIRS+=("$tmp")
  if ! kaggle datasets download "$SLUG" -p "$tmp" >/dev/null 2>&1; then
    echo "[persist] no dataset '$SLUG' yet (first run?); skipping restore"
    return 0
  fi
  local outer
  outer="$(find "$tmp" -name '*.zip' | head -1)"
  if [ -z "$outer" ]; then
    echo "[persist] dataset '$SLUG' download contained no archive; skipping restore"
    return 0
  fi
  mkdir -p "$tmp/dl"
  unzip -q -o "$outer" -d "$tmp/dl"
  if [ ! -f "$tmp/dl/run_state.zip" ]; then
    echo "[persist] dataset '$SLUG' has no run_state.zip; skipping restore"
    return 0
  fi
  # Unpack guard: extracting run_state.zip needs its full unpacked size.
  local need_bytes unpack_kb
  need_bytes="$(python -c 'import sys, zipfile; print(sum(i.file_size for i in zipfile.ZipFile(sys.argv[1]).infolist()))' "$(wpath "$tmp/dl/run_state.zip")")"
  unpack_kb=$((need_bytes / 1024 + 524288))
  kb="$(free_kb)"
  if [ "$kb" -lt "$unpack_kb" ]; then
    echo "[persist] only $((kb / 1024 / 1024)) GB free, need ~$((unpack_kb / 1024 / 1024)) GB to unpack; skipping restore"
    return 0
  fi
  mkdir -p "$tmp/run"
  unzip -q -o "$tmp/dl/run_state.zip" -d "$tmp/run"
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
  local iter run_bytes need_kb kb upload_dir
  iter="$(python -c "import json; print(json.load(open('$RUN_DIR/run_state.json'))['iteration'])")"
  run_bytes="$(du -sb "$RUN_DIR" | awk '{print $1}')"
  # The run dir plus a same-size zip of it must both fit, plus 1 GB slack.
  need_kb=$((run_bytes / 512 + 1048576))
  kb="$(free_kb)"
  if [ "$kb" -lt "$need_kb" ]; then
    echo "[persist] WARN: $((kb / 1024 / 1024)) GB free, need ~$((need_kb / 1024 / 1024)) GB to package; skipping save"
    return 0
  fi
  echo "[persist] packaging run state: $((run_bytes / 1024 / 1024)) MB, $((kb / 1024 / 1024)) GB free"
  upload_dir="$(mktemp -d)"
  CLEANUP_DIRS+=("$upload_dir")
  rm -f "$RUN_DIR/dataset-metadata.json" # stale file from older save attempts
  python - "$RUN_DIR" "$(wpath "$upload_dir/run_state.zip")" <<'PY'
import os
import sys
import zipfile

root, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            arc = os.path.relpath(path, root).replace(os.sep, "/")
            zf.write(path, arc)
PY
  printf '{"title":"ChessAI clean AZ run state","id":"%s","licenses":[{"name":"CC BY-SA 4.0"}]}\n' \
    "$SLUG" > "$upload_dir/dataset-metadata.json"
  if kaggle datasets metadata "$SLUG" >/dev/null 2>&1; then
    # -d keeps only the newest version so Kaggle-side storage stays ~1x
    # the run size (flag exists in both kaggle 2.0.2 and 2.2.2).
    kaggle datasets version -p "$upload_dir" -m "iteration $iter auto-save" -d
  else
    # NOTE: `datasets create` has no -m flag in any kaggle release; the
    # initial message lives in dataset-metadata.json only.
    kaggle datasets create -p "$upload_dir" --public
  fi
  echo "[persist] saved run state (iteration $iter) to https://www.kaggle.com/datasets/$SLUG"
}

case "$ACTION" in
  restore) restore_run ;;
  save) save_run ;;
esac
