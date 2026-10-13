"""Zip the run's checkpoints and upload the archive to a Kaggle Dataset.

Manual one-shot export (from a live Kaggle session that still has run state):

    python tools/upload_checkpoints.py
    python tools/upload_checkpoints.py --dry-run     # build the zip only
    python tools/upload_checkpoints.py --prune-old   # also drop old versions

Steps:
  1. resolve Kaggle credentials: environment variables, then
     ~/.kaggle/kaggle.json, then Kaggle notebook secrets (Add-ons -> Secrets)
  2. pack <run_dir>/checkpoints plus manifest.json and run_state.json into a
     single checkpoints.zip (replay stays out: the run-state dataset owns
     session resume)
  3. upload it as a new version of the target dataset (created if missing)

Python twin of scripts/kaggle_upload_checkpoints.sh, for exporting
checkpoints by hand (e.g. before stopping a session).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from az.config import RunConfig  # noqa: E402

DEFAULT_SLUG = "arjunthakur9999/checkpoints"


def resolve_credentials() -> None:
    """Export KAGGLE_USERNAME/KAGGLE_KEY into os.environ, or raise SystemExit."""

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("[creds] using environment variables")
        return
    home = os.environ.get("HOME") or str(Path.home())
    config_path = Path(home) / ".kaggle" / "kaggle.json"
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            username, key = data["username"], data["key"]
        except (OSError, ValueError, KeyError, TypeError):
            username, key = "", ""
        if username and key:
            os.environ["KAGGLE_USERNAME"] = str(username)
            os.environ["KAGGLE_KEY"] = str(key)
            print(f"[creds] loaded from {config_path}")
            return
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        username = secrets.get_secret("KAGGLE_USERNAME")
        key = secrets.get_secret("KAGGLE_KEY")
    except Exception:
        username, key = "", ""
    if username and key:
        os.environ["KAGGLE_USERNAME"] = str(username)
        os.environ["KAGGLE_KEY"] = str(key)
        print("[creds] loaded from Kaggle notebook secrets")
        return
    raise SystemExit("ERROR: no Kaggle credentials (env, kaggle.json, or Kaggle secrets)")


def build_zip(run_dir: Path, out: Path) -> int:
    """Pack provenance files plus every file under checkpoints/ into `out`.

    Returns the number of files stored.  Replay buffers are intentionally
    excluded: they belong to the run-state dataset used for session resume.
    """

    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("manifest.json", "run_state.json"):
            path = run_dir / name
            if path.is_file():
                archive.write(path, name)
                count += 1
        checkpoints = run_dir / "checkpoints"
        if checkpoints.is_dir():
            for path in sorted(checkpoints.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir).as_posix())
                    count += 1
    return count


def _iteration(run_dir: Path) -> str:
    state = run_dir / "run_state.json"
    if not state.is_file():
        return "manual"
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "manual"
    return str(data.get("iteration", "manual"))


def _resolve_kaggle_cli() -> str:
    """Path to the kaggle CLI, installing the package on first use.

    A resolved absolute path matters on Windows, where subprocess does not
    apply PATHEXT to a bare command name.
    """

    resolved = shutil.which("kaggle")
    if resolved is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "kaggle"],
            check=True,
        )
        resolved = shutil.which("kaggle")
    return resolved or "kaggle"


def _write_metadata(upload_dir: Path, slug: str) -> None:
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "ChessAI clean AZ checkpoints",
                "id": slug,
                "licenses": [{"name": "CC BY-SA 4.0"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _ensure_metadata_identity(path: Path, slug: str) -> None:
    """Guarantee dataset-metadata.json exists and identifies the dataset.

    kaggle CLI 2.x `datasets version`/`create` fail with "ID or slug must be
    specified in the metadata" unless an `id` or `id_no` key is present, and
    the file downloaded by `datasets metadata` does not always carry one.
    Existing fields (title, description, ...) are preserved; only a missing
    identity is filled in, and an absent/unreadable file becomes the
    minimal metadata.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("metadata is not a JSON object")
    except (OSError, ValueError):
        _write_metadata(path.parent, slug)
        return
    if not data.get("id") and not data.get("id_no"):
        data["id"] = slug
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _upload(upload_dir: Path, slug: str, message: str, prune_old: bool, cli: str) -> None:
    probe = subprocess.run(
        [cli, "datasets", "metadata", slug, "-p", str(upload_dir)],
        capture_output=True,
        text=True,
    )
    # Keep whatever the probe downloaded (it holds the dataset's real
    # title/description), but make sure it identifies the dataset; a missing
    # or unreadable file becomes the minimal metadata.  Old versions are
    # kept unless prune_old, so files already in the dataset are never
    # destroyed while the merge/replace semantics are still unverified.
    _ensure_metadata_identity(upload_dir / "dataset-metadata.json", slug)
    if probe.returncode == 0:
        command = [cli, "datasets", "version", "-p", str(upload_dir), "-m", message]
        if prune_old:
            command.append("-d")
    else:
        command = [cli, "datasets", "create", "-p", str(upload_dir)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(Path(ROOT) / "configs" / "clean_az.json"),
        help="Path to clean run JSON config (default: configs/clean_az.json).",
    )
    parser.add_argument(
        "--slug",
        default=DEFAULT_SLUG,
        help=f"Target dataset slug (default: {DEFAULT_SLUG}).",
    )
    parser.add_argument(
        "--prune-old",
        action="store_true",
        help="Delete older dataset versions after a successful upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the zip and report, without needing credentials or uploading.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    os.chdir(ROOT)  # run_root is cwd-relative, exactly like the bash scripts
    run_dir = RunConfig.load_json(config_path).root
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir() or not any(checkpoints.glob("*.pt")):
        raise SystemExit(f"ERROR: no .pt checkpoints under {checkpoints} yet")

    with tempfile.TemporaryDirectory() as tmp:
        upload_dir = Path(tmp)
        zip_path = upload_dir / "checkpoints.zip"
        count = build_zip(run_dir, zip_path)
        iteration = _iteration(run_dir)
        size_mb = zip_path.stat().st_size // (1024 * 1024)
        print(f"[upload] packaged {count} files, {size_mb} MB (iteration {iteration})")

        if args.dry_run:
            with zipfile.ZipFile(zip_path) as archive:
                for name in sorted(archive.namelist()):
                    print(f"  {name}")
            print(f"[upload] dry run; nothing sent to {args.slug}")
            return 0

        resolve_credentials()
        cli = _resolve_kaggle_cli()
        try:
            _upload(upload_dir, args.slug, f"iteration {iteration} checkpoints", args.prune_old, cli)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"ERROR: kaggle upload failed (exit {error.returncode})"
            ) from error

    print(f"[upload] saved checkpoints (iteration {iteration}) to https://www.kaggle.com/datasets/{args.slug}")
    listing = subprocess.run(
        [cli, "datasets", "files", args.slug],
        capture_output=True,
        text=True,
    )
    if listing.returncode == 0 and listing.stdout.strip():
        print(listing.stdout.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
