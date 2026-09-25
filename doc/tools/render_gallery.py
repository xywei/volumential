#!/usr/bin/env python3
"""Regenerate the static gallery assets from maintained Volumential examples.

Sphinx never imports or runs this tool; documentation builds only consume the
files already present under doc/source/_static/gallery.

The numerical targets run OpenCL examples. They need an explicit PyOpenCL
context selector (--pyopencl-ctx or PYOPENCL_CTX); the tool does not let
PyOpenCL pick a device on its own. Each example writes into a scratch directory
(build/gallery-work/<mode>/<target>/ by default), and only the curated figures
are copied into <output-dir>/<target>/. Outputs from a previous run are deleted
first, a missing figure is an error, and manifest.json beside the assets records
how every target was produced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "doc" / "source" / "_static" / "gallery"
_DEFAULT_WORK = _REPO_ROOT / "build" / "gallery-work"
_CONCEPT_SOURCE = _REPO_ROOT / "doc" / "gallery-src"
_RENDERER = "doc/tools/render_gallery.py"
_MANIFEST_NAME = "manifest.json"
_MANIFEST_FORMAT = 1


@dataclass(frozen=True)
class _Example:
    """How to run one maintained example and which figures it contributes."""

    script: str
    figures: tuple[str, ...]
    #: Environment variable naming the example's output directory.
    output_env: str | None = None
    #: Command-line option naming the example's output directory.
    output_option: str | None = None
    #: Command-line flag selecting smoke settings; without one, the example
    #: reads ``VOLUMENTIAL_EXAMPLE_SMOKE``.
    smoke_flag: str | None = None


_EXAMPLES = {
    "laplace2d": _Example(
        script="examples/laplace2d.py",
        figures=("laplace2d_overview.svg", "laplace2d_tree.svg"),
        output_env="VOLUMENTIAL_GALLERY_OUTPUT_DIR",
    ),
    "poisson3d": _Example(
        script="examples/poisson3d.py",
        # The example also writes poisson3d_error_point_cloud.png. At full
        # settings it is a large, low-contrast cloud of dots that adds little
        # to the slices, so it stays in the work directory.
        figures=("poisson3d_slices.png",),
        output_env="VOLUMENTIAL_POISSON3D_OUTPUT_DIR",
    ),
    "branched-flow": _Example(
        script="examples/branched_flow_helmholtz2d.py",
        figures=("branched_flow.png",),
        output_option="--output-dir",
        smoke_flag="--smoke",
    ),
}
_TARGETS = ("concepts", *_EXAMPLES)

# Run in the examples' interpreter and environment. ``-P`` keeps the working
# directory off sys.path, so this reports the volumential the examples import.
_VERSION_PROBE = """\
import json, platform
import matplotlib, numpy, pyopencl, volumential
print(json.dumps({
    "matplotlib": matplotlib.__version__,
    "numpy": numpy.__version__,
    "pyopencl": pyopencl.VERSION_TEXT,
    "python": platform.python_version(),
    "volumential": volumential.volumential_version,
}))
"""


class _Paths:
    """Spell paths for the manifest without revealing absolute locations."""

    def __init__(self, output_dir, work_dir):
        self.output_dir = output_dir
        self.work_dir = work_dir

    def show(self, path):
        path = path.resolve()
        for anchor, label in (
            (_REPO_ROOT, None),
            (self.output_dir, "<output-dir>"),
            (self.work_dir, "<work-dir>"),
        ):
            if path.is_relative_to(anchor):
                relative = path.relative_to(anchor).as_posix()
                return relative if label is None else f"{label}/{relative}"
        return f"<outside-repository>/{path.name}"


def _git(*args):
    return subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _source_state(output_dir):
    """Return the checked-out revision and whether tracked files differ from it.

    Untracked files are ignored, as ``git describe --dirty`` does, and so is the
    output directory: regenerated assets do not make their own source dirty.
    """
    pathspec = ["."]
    if output_dir.is_relative_to(_REPO_ROOT) and output_dir != _REPO_ROOT:
        pathspec.append(
            f":(exclude){output_dir.relative_to(_REPO_ROOT).as_posix()}"
        )
    try:
        revision = _git("rev-parse", "HEAD").strip()
        status = _git(
            "status", "--porcelain", "--untracked-files=no", "--", *pathspec
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "render_gallery.py must run in a Git checkout of Volumential so that "
            "the manifest can record the source revision"
        ) from exc
    return {"revision": revision, "dirty": bool(status.strip())}


def _child_environment(settings):
    env = os.environ.copy()
    for key, value in settings.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _probe_versions(env):
    result = subprocess.run(
        [sys.executable, "-P", "-c", _VERSION_PROBE],
        cwd=_REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "cannot import the packages the numerical gallery needs "
            "(volumential, pyopencl, numpy, matplotlib):\n" + result.stderr
        )
    return json.loads(result.stdout)


def _load_manifest(output_dir):
    path = output_dir / _MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != _MANIFEST_FORMAT
        or not isinstance(manifest.get("targets"), dict)
    ):
        manifest = {}
    manifest["format"] = _MANIFEST_FORMAT
    manifest["generator"] = _RENDERER
    manifest.setdefault("targets", {})
    return manifest


def _write_manifest(output_dir, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {path}")


def _delete_previous(output_dir, relpaths):
    """Delete the files a previous run recorded, if they lie in *output_dir*."""
    for relpath in relpaths:
        path = (output_dir / relpath).resolve()
        if path.is_relative_to(output_dir):
            path.unlink(missing_ok=True)


def _render_concepts(output_dir, previous, source_state):
    _delete_previous(output_dir, previous.get("outputs", ()))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source in sorted(_CONCEPT_SOURCE.glob("*.svg")):
        shutil.copyfile(source, output_dir / source.name)
        print(f"Wrote {output_dir / source.name}")
        outputs.append(source.name)
    if not outputs:
        raise SystemExit(f"no concept sources found in {_CONCEPT_SOURCE}")
    return {
        **source_state,
        "regenerate": ["python", _RENDERER, "concepts"],
        "sources": _CONCEPT_SOURCE.relative_to(_REPO_ROOT).as_posix(),
        "outputs": outputs,
    }


def _render_example(name, *, output_dir, work_dir, previous, context):
    example = _EXAMPLES[name]
    paths = context["paths"]
    smoke = context["mode"] == "smoke"
    # One scratch directory per mode: branched_flow_helmholtz2d.py keeps its
    # table cache in its output directory, and a smoke table cannot serve a
    # full run (the root box differs).
    example_work = work_dir / context["mode"] / name
    gallery_dir = output_dir / name

    _delete_previous(output_dir, previous.get("outputs", ()))
    for figure in example.figures:
        (gallery_dir / figure).unlink(missing_ok=True)
        (example_work / figure).unlink(missing_ok=True)
    example_work.mkdir(parents=True, exist_ok=True)

    settings = dict(context["environment"])
    settings["VOLUMENTIAL_EXAMPLE_SMOKE"] = (
        "1" if smoke and example.smoke_flag is None else None
    )
    shown_settings = dict(settings)
    args, shown_args = [], []
    if example.output_env is not None:
        settings[example.output_env] = str(example_work)
        shown_settings[example.output_env] = paths.show(example_work)
    if example.output_option is not None:
        args += [example.output_option, str(example_work)]
        shown_args += [example.output_option, paths.show(example_work)]
    if smoke and example.smoke_flag is not None:
        args.append(example.smoke_flag)
        shown_args.append(example.smoke_flag)

    command = [sys.executable, str(_REPO_ROOT / example.script), *args]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=_child_environment(settings),
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{example.script} exited with status {result.returncode}")

    missing = [f for f in example.figures if not (example_work / f).is_file()]
    if missing:
        raise SystemExit(
            f"{example.script} did not write the expected figures: "
            + ", ".join(missing)
        )
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for figure in example.figures:
        shutil.copyfile(example_work / figure, gallery_dir / figure)
        print(f"Wrote {gallery_dir / figure}")

    regenerate = [
        "python", _RENDERER, name, "--pyopencl-ctx", context["pyopencl_ctx"]
    ]
    if not smoke:
        regenerate.append("--full")
    return {
        **context["source_state"],
        "mode": context["mode"],
        "pyopencl_ctx": context["pyopencl_ctx"],
        "regenerate": regenerate,
        "command": ["python", example.script, *shown_args],
        "environment": shown_settings,
        "outputs": [f"{name}/{figure}" for figure in example.figures],
        "versions": context["versions"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        choices=(*_TARGETS, "all"),
        help="asset group to regenerate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="gallery root (default: doc/source/_static/gallery)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_DEFAULT_WORK,
        help="scratch directory for the examples' full output, one "
        "subdirectory per mode (default: build/gallery-work)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use full example settings instead of smoke/reduced settings",
    )
    parser.add_argument(
        "--pyopencl-ctx",
        default=os.environ.get("PYOPENCL_CTX"),
        help="PyOpenCL context selector passed to the examples as PYOPENCL_CTX "
        "(default: the PYOPENCL_CTX environment variable)",
    )
    arguments = parser.parse_args()

    targets = _TARGETS if arguments.target == "all" else (arguments.target,)
    output_dir = arguments.output_dir.resolve()
    work_dir = arguments.work_dir.resolve()
    source_state = _source_state(output_dir)
    numerical = [target for target in targets if target in _EXAMPLES]

    context = None
    if numerical:
        if not arguments.pyopencl_ctx:
            parser.error(
                "set PYOPENCL_CTX or pass --pyopencl-ctx (for example "
                "portable:0); the gallery renderer does not choose an OpenCL "
                "device implicitly"
            )
        environment = {
            "PYOPENCL_CTX": arguments.pyopencl_ctx,
            # pyopencl.create_some_context prefers PYOPENCL_TEST when it is set.
            "PYOPENCL_TEST": None,
            "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg",
        }
        context = {
            "mode": "full" if arguments.full else "smoke",
            "pyopencl_ctx": arguments.pyopencl_ctx,
            "environment": environment,
            "paths": _Paths(output_dir, work_dir),
            "source_state": source_state,
            "versions": _probe_versions(_child_environment(environment)),
        }

    manifest = _load_manifest(output_dir)
    try:
        for target in targets:
            previous = manifest["targets"].pop(target, {})
            if target == "concepts":
                record = _render_concepts(output_dir, previous, source_state)
            else:
                record = _render_example(
                    target,
                    output_dir=output_dir,
                    work_dir=work_dir,
                    previous=previous,
                    context=context,
                )
            manifest["targets"][target] = record
    finally:
        _write_manifest(output_dir, manifest)


if __name__ == "__main__":
    main()
