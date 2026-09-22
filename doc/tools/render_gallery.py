#!/usr/bin/env python3
"""Regenerate static gallery assets from maintained Volumential examples.

This tool is deliberately not imported or invoked by Sphinx. Numerical gallery
assets require a working OpenCL runtime; documentation builds only consume the
files already present under doc/source/_static/gallery.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "doc" / "source" / "_static" / "gallery"


def _concept_near_far_svg():
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 960 560\" role=\"img\" aria-labelledby=\"title desc\">\n  <title id=\"title\">Near-field and far-field split in Volumential</title>\n  <desc id=\"desc\">A target box and its neighboring boxes are evaluated from a near-field interaction table, while more distant boxes are handled by the fast multipole method.</desc>\n  <rect width=\"960\" height=\"560\" rx=\"24\" fill=\"#ffffff\"/>\n  <text x=\"48\" y=\"58\" font-family=\"system-ui, sans-serif\" font-size=\"30\" font-weight=\"700\" fill=\"#172033\">One target, two numerical paths</text>\n  <text x=\"48\" y=\"90\" font-family=\"system-ui, sans-serif\" font-size=\"17\" fill=\"#556070\">The singular neighborhood is tabulated; everything farther away stays in the ordinary FMM.</text>\n  <g transform=\"translate(68 130)\" stroke=\"#c7ced8\" stroke-width=\"2\">\n    <rect x=\"0\" y=\"0\" width=\"420\" height=\"420\" rx=\"14\" fill=\"#f7f8fa\" stroke=\"none\"/>\n    <g fill=\"#edf1f5\">\n      <rect x=\"0\" y=\"0\" width=\"84\" height=\"84\"/><rect x=\"84\" y=\"0\" width=\"84\" height=\"84\"/><rect x=\"168\" y=\"0\" width=\"84\" height=\"84\"/><rect x=\"252\" y=\"0\" width=\"84\" height=\"84\"/><rect x=\"336\" y=\"0\" width=\"84\" height=\"84\"/>\n      <rect x=\"0\" y=\"84\" width=\"84\" height=\"84\"/><rect x=\"336\" y=\"84\" width=\"84\" height=\"84\"/>\n      <rect x=\"0\" y=\"168\" width=\"84\" height=\"84\"/><rect x=\"336\" y=\"168\" width=\"84\" height=\"84\"/>\n      <rect x=\"0\" y=\"252\" width=\"84\" height=\"84\"/><rect x=\"336\" y=\"252\" width=\"84\" height=\"84\"/>\n      <rect x=\"0\" y=\"336\" width=\"84\" height=\"84\"/><rect x=\"84\" y=\"336\" width=\"84\" height=\"84\"/><rect x=\"168\" y=\"336\" width=\"84\" height=\"84\"/><rect x=\"252\" y=\"336\" width=\"84\" height=\"84\"/><rect x=\"336\" y=\"336\" width=\"84\" height=\"84\"/>\n    </g>\n    <g fill=\"#fff1cc\">\n      <rect x=\"84\" y=\"84\" width=\"84\" height=\"84\"/><rect x=\"168\" y=\"84\" width=\"84\" height=\"84\"/><rect x=\"252\" y=\"84\" width=\"84\" height=\"84\"/>\n      <rect x=\"84\" y=\"168\" width=\"84\" height=\"84\"/><rect x=\"252\" y=\"168\" width=\"84\" height=\"84\"/>\n      <rect x=\"84\" y=\"252\" width=\"84\" height=\"84\"/><rect x=\"168\" y=\"252\" width=\"84\" height=\"84\"/><rect x=\"252\" y=\"252\" width=\"84\" height=\"84\"/>\n    </g>\n    <rect x=\"168\" y=\"168\" width=\"84\" height=\"84\" rx=\"8\" fill=\"#5a67d8\" stroke=\"#3944aa\" stroke-width=\"3\"/>\n    <g fill=\"none\">\n      <path d=\"M84 0v420M168 0v420M252 0v420M336 0v420M0 84h420M0 168h420M0 252h420M0 336h420\"/>\n    </g>\n    <text x=\"210\" y=\"205\" text-anchor=\"middle\" font-family=\"system-ui, sans-serif\" font-size=\"16\" font-weight=\"700\" fill=\"#ffffff\">target</text>\n    <text x=\"210\" y=\"227\" text-anchor=\"middle\" font-family=\"system-ui, sans-serif\" font-size=\"13\" fill=\"#eef0ff\">box</text>\n  </g>\n  <g font-family=\"system-ui, sans-serif\">\n    <path d=\"M470 235 C555 220 575 184 630 170\" fill=\"none\" stroke=\"#d89b18\" stroke-width=\"4\"/>\n    <polygon points=\"630,170 614,162 617,181\" fill=\"#d89b18\"/>\n    <text x=\"650\" y=\"155\" font-size=\"22\" font-weight=\"700\" fill=\"#9a6b08\">Near field</text>\n    <text x=\"650\" y=\"184\" font-size=\"16\" fill=\"#556070\">same / neighboring leaf boxes</text>\n    <rect x=\"650\" y=\"204\" width=\"246\" height=\"74\" rx=\"14\" fill=\"#fff7df\" stroke=\"#e3b547\" stroke-width=\"2\"/>\n    <text x=\"773\" y=\"235\" text-anchor=\"middle\" font-size=\"17\" font-weight=\"700\" fill=\"#6c4a05\">interaction table</text>\n    <text x=\"773\" y=\"258\" text-anchor=\"middle\" font-size=\"14\" fill=\"#6c4a05\">singular-aware quadrature, reused</text>\n    <path d=\"M470 388 C555 405 575 427 630 438\" fill=\"none\" stroke=\"#778391\" stroke-width=\"4\"/>\n    <polygon points=\"630,438 615,428 613,447\" fill=\"#778391\"/>\n    <text x=\"650\" y=\"388\" font-size=\"22\" font-weight=\"700\" fill=\"#3b4652\">Far field</text>\n    <text x=\"650\" y=\"417\" font-size=\"16\" fill=\"#556070\">well-separated boxes</text>\n    <rect x=\"650\" y=\"454\" width=\"246\" height=\"62\" rx=\"14\" fill=\"#f1f4f7\" stroke=\"#b5bec8\" stroke-width=\"2\"/>\n    <text x=\"773\" y=\"492\" text-anchor=\"middle\" font-size=\"17\" font-weight=\"700\" fill=\"#35404b\">multipole / local expansions</text>\n  </g>\n  <g font-family=\"system-ui, sans-serif\" font-size=\"14\">\n    <rect x=\"68\" y=\"520\" width=\"18\" height=\"18\" rx=\"3\" fill=\"#5a67d8\"/><text x=\"94\" y=\"534\" fill=\"#556070\">target</text>\n    <rect x=\"158\" y=\"520\" width=\"18\" height=\"18\" rx=\"3\" fill=\"#fff1cc\" stroke=\"#e3b547\"/><text x=\"184\" y=\"534\" fill=\"#556070\">table lookup</text>\n    <rect x=\"286\" y=\"520\" width=\"18\" height=\"18\" rx=\"3\" fill=\"#edf1f5\" stroke=\"#c7ced8\"/><text x=\"312\" y=\"534\" fill=\"#556070\">FMM</text>\n  </g>\n</svg>\n"


def _concept_workflow_svg():
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1120 390\" role=\"img\" aria-labelledby=\"title desc\">\n  <title id=\"title\">Volumential volume FMM workflow</title>\n  <desc id=\"desc\">Source density is sampled on volume quadrature nodes, organized into a tree, split into a near-field table path and far-field FMM path, then accumulated into the output potential.</desc>\n  <rect width=\"1120\" height=\"390\" rx=\"24\" fill=\"#ffffff\"/>\n  <text x=\"48\" y=\"54\" font-family=\"system-ui, sans-serif\" font-size=\"30\" font-weight=\"700\" fill=\"#172033\">From a source density to a volume potential</text>\n  <text x=\"48\" y=\"84\" font-family=\"system-ui, sans-serif\" font-size=\"17\" fill=\"#556070\">The special part is local: only near interactions leave the ordinary particle FMM path.</text>\n  <g font-family=\"system-ui, sans-serif\">\n    <rect x=\"50\" y=\"132\" width=\"180\" height=\"112\" rx=\"18\" fill=\"#eef4ff\" stroke=\"#8ba8e8\" stroke-width=\"2\"/>\n    <text x=\"140\" y=\"168\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#27457f\">Source density</text>\n    <text x=\"140\" y=\"194\" text-anchor=\"middle\" font-size=\"15\" fill=\"#52698f\">f(y)</text>\n    <text x=\"140\" y=\"218\" text-anchor=\"middle\" font-size=\"14\" fill=\"#52698f\">box quadrature nodes</text>\n    <rect x=\"300\" y=\"132\" width=\"180\" height=\"112\" rx=\"18\" fill=\"#f3f1ff\" stroke=\"#a399df\" stroke-width=\"2\"/>\n    <text x=\"390\" y=\"168\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#4c438c\">Tree + traversal</text>\n    <text x=\"390\" y=\"194\" text-anchor=\"middle\" font-size=\"14\" fill=\"#625a90\">adaptive 2:1 boxes</text>\n    <text x=\"390\" y=\"218\" text-anchor=\"middle\" font-size=\"14\" fill=\"#625a90\">interaction lists</text>\n    <rect x=\"565\" y=\"112\" width=\"206\" height=\"92\" rx=\"18\" fill=\"#fff7df\" stroke=\"#e3b547\" stroke-width=\"2\"/>\n    <text x=\"668\" y=\"148\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#7a5407\">Near field</text>\n    <text x=\"668\" y=\"174\" text-anchor=\"middle\" font-size=\"14\" fill=\"#7a650e\">precomputed table lookup</text>\n    <rect x=\"565\" y=\"226\" width=\"206\" height=\"92\" rx=\"18\" fill=\"#f1f4f7\" stroke=\"#b5bec8\" stroke-width=\"2\"/>\n    <text x=\"668\" y=\"262\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#35404b\">Far field</text>\n    <text x=\"668\" y=\"288\" text-anchor=\"middle\" font-size=\"14\" fill=\"#556070\">ordinary FMM expansions</text>\n    <rect x=\"850\" y=\"160\" width=\"220\" height=\"112\" rx=\"18\" fill=\"#eaf8f1\" stroke=\"#72bd96\" stroke-width=\"2\"/>\n    <text x=\"960\" y=\"196\" text-anchor=\"middle\" font-size=\"18\" font-weight=\"700\" fill=\"#216b48\">Potential</text>\n    <text x=\"960\" y=\"222\" text-anchor=\"middle\" font-size=\"15\" fill=\"#3e765e\">near + far contributions</text>\n    <text x=\"960\" y=\"246\" text-anchor=\"middle\" font-size=\"14\" fill=\"#3e765e\">at quadrature / target points</text>\n    <g fill=\"none\" stroke=\"#7b8794\" stroke-width=\"4\">\n      <path d=\"M230 188h56\"/><path d=\"M480 188h54\"/><path d=\"M480 188 C520 188 520 158 550 158\"/>\n      <path d=\"M480 188 C520 188 520 272 550 272\"/>\n      <path d=\"M771 158 C814 158 814 205 837 205\"/>\n      <path d=\"M771 272 C814 272 814 227 837 227\"/>\n    </g>\n    <g fill=\"#7b8794\">\n      <polygon points=\"286,188 273,180 273,196\"/><polygon points=\"550,158 537,150 537,166\"/>\n      <polygon points=\"550,272 537,264 537,280\"/><polygon points=\"837,205 824,197 824,213\"/>\n      <polygon points=\"837,227 824,219 824,235\"/>\n    </g>\n  </g>\n  <rect x=\"300\" y=\"338\" width=\"470\" height=\"34\" rx=\"17\" fill=\"#f8fafc\"/>\n  <text x=\"535\" y=\"360\" text-anchor=\"middle\" font-family=\"system-ui, sans-serif\" font-size=\"14\" fill=\"#556070\">Near-field tables depend on geometry/kernel/order — not on the source density.</text>\n</svg>\n"


def _write_concepts(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "near-far-anatomy.svg": _concept_near_far_svg(),
        "volume-fmm-workflow.svg": _concept_workflow_svg(),
    }
    for filename, contents in outputs.items():
        path = output_dir / filename
        path.write_text(contents, encoding="utf-8")
        print(f"Wrote {path}")


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
        choices=("concepts", "laplace2d", "poisson3d", "branched-flow", "all"),
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
