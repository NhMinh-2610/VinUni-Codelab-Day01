"""Load the submitted lab consistently for every offline test module.

Set LAB_SOLUTION_PATH to a Python file inside this repository to test a specific
copy. Without an override, prefer the submission, then the reference solution,
then the starter file.
"""

import importlib.util
import os
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parents[1]


def resolve_target_path() -> Path:
    override = os.getenv("LAB_SOLUTION_PATH", "").strip()
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = LAB_DIR / path
        path = path.resolve()
        if not path.is_relative_to(LAB_DIR):
            raise ValueError("LAB_SOLUTION_PATH must point to a file inside the lab repository.")
        if path.suffix != ".py":
            raise ValueError("LAB_SOLUTION_PATH must point to a Python (.py) file.")
        if not path.is_file():
            raise FileNotFoundError(f"LAB_SOLUTION_PATH does not exist: {path}")
        return path

    for relative in ("solution/solution.py", "solution-code/solution.py", "starter-code/template.py"):
        path = LAB_DIR / relative
        if path.is_file():
            return path
    raise FileNotFoundError("No solution/solution.py, solution-code/solution.py, or starter template found.")


def load_target_module(unique_name: str):
    path = resolve_target_path()
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load lab module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module
