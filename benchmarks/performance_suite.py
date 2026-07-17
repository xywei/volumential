#!/usr/bin/env python3
"""Run the maintained performance benchmark suite.

This wrapper standardizes the benchmark commands used for scaling, cache,
parameter, and adaptive timing evidence. Smoke mode is intended for quick local
or CI checks. Full mode is intended for a controlled remote compute host and
should be wrapped by the paper repository's metadata capture tool when results
are promoted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteCase:
    name: str
    description: str
    script: str
    output_name: str | None
    accepts_backend: bool = False
    arrays_output_name: str | None = None
    metadata_output_name: str | None = None


SUITE_CASES = (
    SuiteCase(
        name="table-equivalence-cache",
        description="canonical table scaling, cold build, warm load, payload size",
        script="benchmarks/table_equivalence_cache.py",
        output_name=None,
    ),
    SuiteCase(
        name="accuracy-preservation",
        description="FMM output preservation for canonical rescaled tables",
        script="benchmarks/accuracy_preservation.py",
        output_name="accuracy_preservation.csv",
    ),
    SuiteCase(
        name="split-parameter-sweep",
        description="Helmholtz/Yukawa split-order and parameter coverage",
        script="benchmarks/split_parameter_sweep.py",
        output_name="split_parameter_sweep.csv",
        accepts_backend=True,
    ),
    SuiteCase(
        name="adaptive-timing",
        description="adaptive-tree geometry, table build/load, and FMM wall time",
        script="benchmarks/adaptive_timing.py",
        output_name="adaptive_timing.csv",
        accepts_backend=True,
    ),
    SuiteCase(
        name="dmk-effective-density",
        description="controlled Gaussian split effective-density diagnostic",
        script="benchmarks/dmk_effective_density.py",
        output_name="dmk_effective_density.csv",
        accepts_backend=True,
        arrays_output_name="dmk_effective_density_arrays.npz",
        metadata_output_name="dmk_effective_density_metadata.json",
    ),
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _case_output_arg(case: SuiteCase, case_dir: Path) -> list[str]:
    args = []
    if case.output_name is None:
        args.extend(["--out-dir", str(case_dir)])
    else:
        args.extend(["--out", str(case_dir / case.output_name)])
    if case.arrays_output_name is not None:
        args.extend(["--arrays-out", str(case_dir / case.arrays_output_name)])
    if case.metadata_output_name is not None:
        args.extend(["--metadata-out", str(case_dir / case.metadata_output_name)])
    return args


def _case_cache_arg(case_dir: Path) -> list[str]:
    return ["--cache-dir", str(case_dir / "cache")]


def _run_case(
    case: SuiteCase, *, mode: str, backend: str, out_dir: Path, dry_run: bool
) -> dict[str, object]:
    case_dir = out_dir / case.name
    script_path = REPO_ROOT / case.script
    command = [
        sys.executable,
        str(script_path),
        "--mode",
        mode,
        *_case_output_arg(case, case_dir),
        *_case_cache_arg(case_dir),
    ]
    if case.accepts_backend:
        command.extend(["--backend", backend])

    record: dict[str, object] = {
        "name": case.name,
        "description": case.description,
        "mode": mode,
        "command": command,
        "backend": backend if case.accepts_backend else None,
        "out_dir": str(case_dir),
        "dry_run": dry_run,
    }

    if dry_run:
        return record

    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(command, check=False, cwd=REPO_ROOT)
    record["elapsed_s"] = time.perf_counter() - start
    record["returncode"] = completed.returncode
    return record


def _write_manifest(records: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"cases": records}, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--backend",
        choices=("auto", "pocl-cpu", "cuda-gpu"),
        default="auto",
        help="OpenCL backend for suite cases that expose --backend",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("build/benchmarks/performance-suite"))
    parser.add_argument("--case", choices=[case.name for case in SUITE_CASES], action="append")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="write commands without running cases")
    parser.add_argument("--list-cases", action="store_true", help="print suite cases and exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    selected = [case for case in SUITE_CASES if args.case is None or case.name in args.case]
    if args.list_cases:
        for case in selected:
            print(f"{case.name}: {case.description}")
        return

    out_dir = args.out_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else out_dir / "manifest.json"

    records = [
        _run_case(
            case,
            mode=args.mode,
            backend=args.backend,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
        for case in selected
    ]
    _write_manifest(records, manifest_path)
    print(f"wrote {manifest_path}")

    failures = [record for record in records if record.get("returncode") not in (None, 0)]
    if failures:
        raise SystemExit(int(failures[0]["returncode"]))


if __name__ == "__main__":
    main()
