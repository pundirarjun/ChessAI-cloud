from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]


def test_all_project_python_files_compile():
    ignored = {"__pycache__", ".venv", "venv", ".pytest_cache"}
    for path in ROOT.rglob("*.py"):
        if any(part in ignored for part in path.parts):
            continue
        py_compile.compile(str(path), doraise=True)
