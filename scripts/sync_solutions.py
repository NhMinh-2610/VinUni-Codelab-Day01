"""Copy the completed starter to both submission paths, or check for drift."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "starter-code" / "template.py"
DESTINATIONS = (ROOT / "solution" / "solution.py", ROOT / "solution-code" / "solution.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if a solution copy is missing or stale.")
    args = parser.parse_args()
    content = SOURCE.read_bytes()
    stale = [path for path in DESTINATIONS if not path.is_file() or path.read_bytes() != content]
    if args.check:
        for path in stale:
            print(f"Out of sync: {path.relative_to(ROOT)}")
        if stale:
            print("Run: python scripts/sync_solutions.py")
            return 1
        print("Both solution copies match starter-code/template.py.")
        return 0
    for path in DESTINATIONS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Updated {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
