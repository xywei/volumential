"""Regenerate the computed documentation gallery from maintained examples.

This script is intentionally *not* part of the Sphinx build.  It runs OpenCL
examples, so maintainers invoke it explicitly on a machine whose device and
software environment they are prepared to record.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "doc" / "source" / "images" / "generated"

EXAMPLES = {
    "laplace2d": {
        "command": [
            sys.executable,
            "examples/laplace2d.py",
        ],
        "env": {
            "VOLUMENTIAL_EXAMPLE_SMOKE": "1",
            "VOLUMENTIAL_LAPLACE2D_OUTPUT_DIR": "{output}/laplace2d",
        },
        "outputs": [
            "laplace2d/laplace2d_overview.png",
            "laplace2d/laplace2d_tree.png",
        ],
    },
    "poisson3d": {
        "command": [
            sys.executable,
            "examples/poisson3d.py",
        ],
        "env": {
            "VOLUMENTIAL_EXAMPLE_SMOKE": "1",
            "VOLUMENTIAL_POISSON3D_OUTPUT_DIR": "{output}/poisson3d",
        },
        "outputs": [
            "poisson3d/poisson3d_slices.png",
            "poisson3d/poisson3d_error_point_cloud.png",
        ],
    },
    "branched-flow": {
        "command": [
            sys.executable,
            "examples/branched_flow_helmholtz2d.py",
            "--smoke",
            "--output-dir",
            "{output}/branched-flow",
        ],
        "env": {},
        "outputs": [
            "branched-flow/branched_flow.png",
        ],
    },
}


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _expand(items: list[str], output_dir: Path) -> list[str]:
    return [item.format(output=str(output_dir)) for item in items]


def _run_example(name: str, output_dir: Path, pyopencl_ctx: str) -> dict:
    spec = EXAMPLES[name]
    command = _expand(spec["command"], output_dir)
    env = os.environ.copy()
    env["PYOPENCL_CTX"] = pyopencl_ctx
    env.update(
        {
            key: value.format(output=str(output_dir))
            for key, value in spec["env"].items()
        }
    )

    outputs = [output_dir / relpath for relpath in spec["outputs"]]
    for output in outputs:
        output.unlink(missing_ok=True)

    print("+", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    missing = [path for path in outputs if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"{name} did not produce expected files: {missing_text}")

    return {
        "name": name,
        "command": command,
        "environment": {
            key: env[key]
            for key in sorted({"PYOPENCL_CTX", *spec["env"].keys()})
        },
        "outputs": [
            str(path.relative_to(output_dir))
            for path in outputs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory receiving generated images and manifest",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(EXAMPLES),
        help="render only this example; repeat to select several",
    )
    parser.add_argument(
        "--pyopencl-ctx",
        default=os.environ.get("PYOPENCL_CTX"),
        help="explicit PyOpenCL context selector; defaults to PYOPENCL_CTX",
    )
    args = parser.parse_args()

    if not args.pyopencl_ctx:
        parser.error(
            "set PYOPENCL_CTX or pass --pyopencl-ctx; gallery generation "
            "must not choose a device implicitly"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(EXAMPLES)
    records = [
        _run_example(name, output_dir, args.pyopencl_ctx)
        for name in selected
    ]

    manifest = {
        "revision": _git_revision(),
        "pyopencl_ctx": args.pyopencl_ctx,
        "examples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Wrote", manifest_path)


if __name__ == "__main__":
    main()
