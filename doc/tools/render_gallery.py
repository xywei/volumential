#!/usr/bin/env python3
"""Regenerate the computed gallery figures from maintained Volumential examples.

Sphinx never imports or runs this tool; documentation builds only consume the
files already present under doc/source/gallery.

Every target runs an OpenCL example. The tool needs an explicit PyOpenCL
context selector (--pyopencl-ctx or PYOPENCL_CTX); it does not let PyOpenCL pick
a device on its own. Each example runs in, and writes into, a scratch directory
(build/gallery-work/<mode>/<target>/ by default), so its table caches and data
files stay there, and only the curated figures are copied into
<output-dir>/<target>/. Outputs from a previous run are deleted first, a
missing figure is an error, and manifest.json beside the figures records how
every target was produced.

The hand-drawn schematics in doc/source/gallery/ are not generated; they are
edited in place.
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
_DEFAULT_OUTPUT = _REPO_ROOT / "doc" / "source" / "gallery"
_DEFAULT_WORK = _REPO_ROOT / "build" / "gallery-work"
_RENDERER = "doc/tools/render_gallery.py"
_MANIFEST_NAME = "manifest.json"
_MANIFEST_FORMAT = 1
#: Inherited variables with this prefix are removed from the examples'
#: environment. Several of them change a computation (poisson3d.py reads its
#: resolution from VOLUMENTIAL_POISSON3D_*, the library reads cache and
#: build switches), and a render must be defined by the renderer alone.
_EXAMPLE_ENV_PREFIX = "VOLUMENTIAL_"


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
    "laplace2d-adaptive": _Example(
        script="examples/laplace2d_adaptive.py",
        figures=("laplace2d_adaptive.svg",),
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

#: Distributions whose versions the manifest records. Besides the plotting
#: stack they are the packages that decide the computed digits: the tree and
#: the expansions (boxtree, sumpy), the generated kernels (loopy, pymbolic),
#: quadrature (modepy), and the far-field backend of branched-flow (pyfmmlib).
_RECORDED_DISTRIBUTIONS = (
    "boxtree",
    "loopy",
    "modepy",
    "pyfmmlib",
    "pymbolic",
    "pytential",
    "sumpy",
)

# Run in the examples' interpreter and environment. ``-P`` keeps the working
# directory off sys.path, so this reports the volumential the examples import.
# The context is built the way the examples build theirs, from PYOPENCL_CTX;
# only the device type is reported, never its name.
_VERSION_PROBE = """\
import json, platform, sys
from importlib import metadata
from pathlib import Path

import matplotlib, numpy, pyopencl, volumential

def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def device_kind(device):
    kinds = [
        name
        for name in ("CPU", "GPU", "ACCELERATOR")
        if device.type & getattr(pyopencl.device_type, name)
    ]
    return "+".join(kinds) or "OTHER"

context = pyopencl.create_some_context(interactive=False)
print(json.dumps({
    "volumential_file": str(Path(volumential.__file__).resolve()),
    "device_types": sorted({device_kind(device) for device in context.devices}),
    "versions": {
        "matplotlib": matplotlib.__version__,
        "numpy": numpy.__version__,
        "pyopencl": pyopencl.VERSION_TEXT,
        "python": platform.python_version(),
        "volumential": volumential.volumential_version,
        **{name: distribution_version(name) for name in sys.argv[1:]},
    },
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

    Untracked files are ignored, as ``git describe --dirty`` does, and so are
    the files the renderer itself writes (every target's figures and the
    manifest), so regenerated figures do not make their own source dirty. Any
    other file in the output directory, such as a schematic, still counts.
    """
    generated = [output_dir / _MANIFEST_NAME] + [
        output_dir / name / figure
        for name, example in _EXAMPLES.items()
        for figure in example.figures
    ]
    pathspec = ["."] + [
        f":(exclude,literal){path.relative_to(_REPO_ROOT).as_posix()}"
        for path in generated
        if path.is_relative_to(_REPO_ROOT)
    ]
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
    """Return the inherited environment without ``VOLUMENTIAL_*``, plus
    *settings*, where a value of ``None`` removes the variable."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_EXAMPLE_ENV_PREFIX)
    }
    for key, value in settings.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _probe_environment(env):
    """Return the package versions and the device type the examples will see.

    Fails unless the examples import volumential from this checkout: the
    manifest's revision would not describe the code that ran otherwise.
    """
    result = subprocess.run(
        [sys.executable, "-P", "-c", _VERSION_PROBE, *_RECORDED_DISTRIBUTIONS],
        cwd=_REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "cannot import the packages the gallery needs (volumential, "
            "pyopencl, numpy, matplotlib) or create the selected OpenCL "
            "context:\n" + result.stderr
        )
    probe = json.loads(result.stdout)
    if not Path(probe["volumential_file"]).is_relative_to(_REPO_ROOT):
        raise SystemExit(
            "the examples would import volumential from outside this checkout, "
            "so the manifest's revision would not describe the code that ran; "
            "install the checkout in editable mode or put it on PYTHONPATH"
        )
    return {
        "device_type": ",".join(probe["device_types"]),
        "versions": probe["versions"],
    }


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
    # Keep the records of the other current targets; drop retired ones.
    manifest["targets"] = {
        name: record
        for name, record in manifest.get("targets", {}).items()
        if name in _EXAMPLES
    }
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


def _render_example(name, *, output_dir, work_dir, previous, context):
    example = _EXAMPLES[name]
    paths = context["paths"]
    smoke = context["mode"] == "smoke"
    # One scratch directory per mode and target. The example runs there, so
    # the near-field tables it caches in its working directory (laplace2d,
    # poisson3d) or output directory (branched-flow) stay out of the
    # repository root, and a smoke table never meets a full run.
    example_work = work_dir / context["mode"] / name
    gallery_dir = output_dir / name

    _delete_previous(output_dir, previous.get("outputs", ()))
    for figure in example.figures:
        (gallery_dir / figure).unlink(missing_ok=True)
        (example_work / figure).unlink(missing_ok=True)
    example_work.mkdir(parents=True, exist_ok=True)

    smoke_setting = "1" if smoke and example.smoke_flag is None else None
    settings = {
        **context["environment"],
        "VOLUMENTIAL_EXAMPLE_SMOKE": smoke_setting,
    }
    shown_settings = {
        **context["shown_environment"],
        "VOLUMENTIAL_EXAMPLE_SMOKE": smoke_setting,
    }
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
        cwd=example_work,
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
    regenerate += context["directory_options"]
    return {
        **context["source_state"],
        "mode": context["mode"],
        "pyopencl_ctx": context["pyopencl_ctx"],
        "device_type": context["device_type"],
        "regenerate": regenerate,
        "command": ["python", example.script, *shown_args],
        "working_directory": paths.show(example_work),
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
        choices=(*_EXAMPLES, "all"),
        help="figure group to regenerate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="gallery root (default: doc/source/gallery)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_DEFAULT_WORK,
        help="scratch directory the examples run in, one subdirectory per mode "
        "and target (default: build/gallery-work)",
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

    if not arguments.pyopencl_ctx:
        parser.error(
            "set PYOPENCL_CTX or pass --pyopencl-ctx (for example "
            "portable:0); the gallery renderer does not choose an OpenCL "
            "device implicitly"
        )

    targets = tuple(_EXAMPLES) if arguments.target == "all" else (arguments.target,)
    output_dir = arguments.output_dir.resolve()
    work_dir = arguments.work_dir.resolve()
    paths = _Paths(output_dir, work_dir)
    # Matplotlib reads a user matplotlibrc from MATPLOTLIBRC or the config
    # directory; an empty config directory of the renderer's own keeps the
    # figures on Matplotlib's defaults plus what the examples set.
    matplotlib_config = work_dir / "matplotlib-config"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    # Matplotlib also reads a matplotlibrc from the working directory, and the
    # scratch directories are reused between runs: refuse to render over one.
    mode = "full" if arguments.full else "smoke"
    stray_rc = [
        path
        for path in (
            matplotlib_config / "matplotlibrc",
            *(work_dir / mode / target / "matplotlibrc" for target in targets),
        )
        if path.is_file()
    ]
    if stray_rc:
        raise SystemExit(
            "remove matplotlibrc from the renderer's work directories, where it "
            "would restyle the figures: " + ", ".join(map(str, stray_rc))
        )
    environment = {
        "PYOPENCL_CTX": arguments.pyopencl_ctx,
        # pyopencl.create_some_context prefers PYOPENCL_TEST when it is set.
        "PYOPENCL_TEST": None,
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
        "MATPLOTLIBRC": None,
        "MPLCONFIGDIR": str(matplotlib_config),
    }
    shown_environment = {
        **environment,
        "MPLCONFIGDIR": paths.show(matplotlib_config),
    }
    probe = _probe_environment(_child_environment(environment))
    # A non-default directory belongs in the recorded regeneration command;
    # one outside the repository is spelled as a placeholder, not a path.
    directory_options = []
    for option, directory, default, placeholder in (
        ("--output-dir", output_dir, _DEFAULT_OUTPUT, "<output-dir>"),
        ("--work-dir", work_dir, _DEFAULT_WORK, "<work-dir>"),
    ):
        if directory != default.resolve():
            directory_options += [
                option,
                directory.relative_to(_REPO_ROOT).as_posix()
                if directory.is_relative_to(_REPO_ROOT)
                else placeholder,
            ]
    context = {
        "mode": mode,
        "pyopencl_ctx": arguments.pyopencl_ctx,
        "environment": environment,
        "shown_environment": shown_environment,
        "paths": paths,
        "source_state": _source_state(output_dir),
        "directory_options": directory_options,
        **probe,
    }

    manifest = _load_manifest(output_dir)
    try:
        for target in targets:
            previous = manifest["targets"].pop(target, {})
            manifest["targets"][target] = _render_example(
                target,
                output_dir=output_dir,
                work_dir=work_dir,
                previous=previous,
                context=context,
            )
    finally:
        _write_manifest(output_dir, manifest)


if __name__ == "__main__":
    main()
