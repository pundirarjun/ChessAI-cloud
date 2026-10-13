#!/usr/bin/env bash
set -euo pipefail

# Upload the run's checkpoints (checkpoints/iteration_*.pt, best_model.pt,
# candidate_model.pt) as a single checkpoints.zip to a Kaggle Dataset so
# model weights survive session death and are directly downloadable.
#
# Intentionally a separate dataset from kaggle_persist.sh's run-state
# dataset: restore picks the first *.zip it finds, so mixing archives in
# one dataset would break session resume.
#
# Usage: kaggle_upload_checkpoints.sh [project_root] [config]
#
# Enabled automatically inside Kaggle notebooks (detected via /kaggle/input
# or KAGGLE_KERNEL_RUN_TYPE); credentials come from KAGGLE_USERNAME/KAGGLE_KEY,
# ~/.kaggle/kaggle.json, or Kaggle notebook secrets (scripts/kaggle_creds.sh).
#   KAGGLE_CHECKPOINT_SLUG     override the dataset slug (default arjunthakur9999/checkpoints)
#   KAGGLE_CHECKPOINT_PRUNE_OLD=1  delete older dataset versions after upload (default: keep)
#   UPLOAD_CHECKPOINTS=0       disable entirely

PROJECT_ROOT="${1:-/kaggle/working/ChessAI-cloud}"
CONFIG="${2:-$PROJECT_ROOT/configs/clean_az.json}"
SLUG="${KAGGLE_CHECKPOINT_SLUG:-arjunthakur9999/checkpoints}"

if [ "${UPLOAD_CHECKPOINTS:-1}" = "0" ]; then
  echo "[upload] disabled via UPLOAD_CHECKPOINTS=0"
  exit 0
fi
if ! { [ -d /kaggle/input ] || [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]; }; then
  echo "[upload] not a Kaggle notebook; skipping checkpoints upload"
  exit 0
fi

cd "$PROJECT_ROOT"

source scripts/kaggle_creds.sh
if ! kaggle_resolve_credentials; then
  echo "ERROR: no Kaggle credentials (env, kaggle.json, or Kaggle secrets)" >&2
  exit 1
fi

command -v kaggle >/dev/null 2>&1 || python -m pip install --quiet kaggle

# tr -d: Python prints CRLF on Windows/Git bash; harmless on Linux.
RUN_DIR="$(python -c "import sys; from az.config import RunConfig; print(RunConfig.load_json(sys.argv[1]).root.as_posix())" "$CONFIG" | tr -d '\r')"

if [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "ERROR: no checkpoints directory at $RUN_DIR/checkpoints yet" >&2
  exit 1
fi
if ! compgen -G "$RUN_DIR/checkpoints/*.pt" >/dev/null; then
  echo "ERROR: no .pt files under $RUN_DIR/checkpoints yet" >&2
  exit 1
fi

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

ITER="manual"
if [ -f "$RUN_DIR/run_state.json" ]; then
  iter_json="$RUN_DIR/run_state.json"
  ITER="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['iteration'])" "$iter_json" | tr -d '\r')"
fi

upload_dir="$(mktemp -d)"
CLEANUP_DIRS+=("$upload_dir")

python - "$RUN_DIR" "$(wpath "$upload_dir/checkpoints.zip")" <<'PY'
import os
import sys
import zipfile

root, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in ("manifest.json", "run_state.json"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            zf.write(path, name)
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "checkpoints")):
        dirnames.sort()
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            arc = os.path.relpath(path, root).replace(os.sep, "/")
            zf.write(path, arc)
PY

size_mb="$(du -sm "$RUN_DIR/checkpoints" | awk '{print $1}')"
echo "[upload] packaging $size_mb MB of checkpoints (iteration $ITER)"

# dataset-metadata.json is required by `datasets version`/`create` and must
# identify the dataset with an `id` or `id_no` (otherwise the CLI fails with
# "ID or slug must be specified in the metadata").  Probe with -p into the
# upload dir so the downloaded file (existing title/description) is kept, and
# fill in only a missing identity; a missing/unreadable file becomes minimal
# metadata.  Old versions are kept unless KAGGLE_CHECKPOINT_PRUNE_OLD=1, so
# files already in the dataset are never destroyed while the merge/replace
# semantics of `datasets version` are still unverified.
ensure_metadata_identity() {
  python - "$upload_dir/dataset-metadata.json" "$SLUG" <<'PY'
import json
import sys

path, slug = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("metadata is not a JSON object")
except Exception:
    data = None
if data is None:
    data = {
        "title": "ChessAI clean AZ checkpoints",
        "id": slug,
        "licenses": [{"name": "CC BY-SA 4.0"}],
    }
elif not data.get("id") and not data.get("id_no"):
    data["id"] = slug
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
}
if kaggle datasets metadata "$SLUG" -p "$upload_dir" >/dev/null 2>&1; then
  ensure_metadata_identity
  if [ "${KAGGLE_CHECKPOINT_PRUNE_OLD:-0}" = "1" ]; then
    kaggle datasets version -p "$upload_dir" -m "iteration $ITER checkpoints" -d
  else
    kaggle datasets version -p "$upload_dir" -m "iteration $ITER checkpoints"
  fi
else
  # Creating: metadata is required, and there is no prior content to lose.
  # NOTE: `datasets create` has no -m flag; the message lives in metadata only.
  ensure_metadata_identity
  kaggle datasets create -p "$upload_dir"
fi
echo "[upload] saved checkpoints (iteration $ITER) to https://www.kaggle.com/datasets/$SLUG"
kaggle datasets files "$SLUG" 2>/dev/null | head -15 || true
