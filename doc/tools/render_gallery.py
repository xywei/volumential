#!/usr/bin/env python3
"""Regenerate static gallery assets from maintained Volumential examples.

This tool is deliberately not imported or invoked by Sphinx. Numerical gallery
assets require a working OpenCL runtime; documentation builds only consume the
files already present under doc/source/_static/gallery.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "doc" / "source" / "_static" / "gallery"
_CONCEPT_SOURCE = _REPO_ROOT / "doc" / "gallery-src"


def _write_concepts(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(_CONCEPT_SOURCE.glob("*.svg")):
        target = output_dir / source.name
        shutil.copyfile(source, target)
        print(f"Wrote {target}")


def _run_example(script, output_dir, *, smoke, args=(), extra_env=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if smoke:
        env["VOLUMENTIAL_EXAMPLE_SMOKE"] = "1"
    else:
        env.pop("VOLUMENTIAL_EXAMPLE_SMOKE", None)
    if extra_env:
        env.update(extra_env)

    command = [sys.executable, str(_REPO_ROOT / script), *args]
    print("+", " ".join(command))
    subprocess.run(command, cwd=_REPO_ROOT, env=env, check=True)


def _render_laplace2d(output_dir, *, smoke):
    _run_example(
        "examples/laplace2d.py",
        output_dir,
        smoke=smoke,
        extra_env={"VOLUMENTIAL_GALLERY_OUTPUT_DIR": str(output_dir)},
    )


def _render_poisson3d(output_dir, *, smoke):
    _run_example(
        "examples/poisson3d.py",
        output_dir,
        smoke=smoke,
        extra_env={"VOLUMENTIAL_POISSON3D_OUTPUT_DIR": str(output_dir)},
    )


def _render_branched_flow(output_dir, *, smoke):
    args = ["--output-dir", str(output_dir)]
    if smoke:
        args.append("--smoke")
    _run_example(
        "examples/branched_flow_helmholtz2d.py",
        output_dir,
        smoke=False,
        args=args,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=(
            "concepts",
            "laplace2d",
            "poisson3d",
            "branched-flow",
            "all",
        ),
        help="asset group to regenerate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="gallery root (default: doc/source/_static/gallery)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use full example settings instead of smoke/reduced settings",
    )
    arguments = parser.parse_args()

    output_dir = arguments.output_dir.resolve()
    smoke = not arguments.full

    if arguments.target in {"concepts", "all"}:
        _write_concepts(output_dir)
    if arguments.target in {"laplace2d", "all"}:
        _render_laplace2d(output_dir / "laplace2d", smoke=smoke)
    if arguments.target in {"poisson3d", "all"}:
        _render_poisson3d(output_dir / "poisson3d", smoke=smoke)
    if arguments.target in {"branched-flow", "all"}:
        _render_branched_flow(output_dir / "branched-flow", smoke=smoke)


if __name__ == "__main__":
    main()
