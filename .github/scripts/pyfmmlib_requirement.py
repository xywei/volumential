"""Print the `pyfmmlib` requirement that `uv.lock` pins, as a direct reference.

`uv pip install` does not read `uv.lock`; only `uv sync` does, and an exact
sync would uninstall the conda-provided half of the CI environment.  So the
`fmmlib` extra's `[tool.uv.sources]` entry, which names a repository but no
revision, would resolve to whatever upstream `main` happens to be on the day
the job runs.

The `Testing (Linux)` job passes what this prints on the `uv pip install`
command line, which pins the build to the locked commit and keeps `uv.lock`
the single place that commit is written down: bump the lock and CI follows.

Usage: python .github/scripts/pyfmmlib_requirement.py [UV_LOCK]
"""

import sys
import tomllib
from pathlib import Path


def pyfmmlib_requirement(lock_path: Path) -> str:
    """Return ``pyfmmlib @ git+<url>@<rev>`` for the entry in ``lock_path``."""

    with lock_path.open("rb") as lock_file:
        lock = tomllib.load(lock_file)

    for package in lock.get("package", []):
        if package.get("name") == "pyfmmlib":
            break
    else:
        raise SystemExit(f"{lock_path}: no pyfmmlib entry")

    source = package.get("source", {}).get("git")
    if source is None:
        raise SystemExit(f"{lock_path}: pyfmmlib is not locked to a Git source")

    # uv writes the resolved commit as the fragment of the Git URL.
    url, _, rev = source.partition("#")
    if not rev:
        raise SystemExit(f"{lock_path}: pyfmmlib pins no revision: {source}")

    return f"pyfmmlib @ git+{url}@{rev}"


def main() -> None:
    """Print the requirement for the lock file named on the command line."""

    lock_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("uv.lock")
    print(pyfmmlib_requirement(lock_path))


if __name__ == "__main__":
    main()
