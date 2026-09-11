# CI and the review bots

Two GitHub Actions workflows and two automated reviewers stand between a branch
and `main`.

## `CI` — every pull request

`.github/workflows/ci.yml` runs on pull requests to `main`, on pushes to
`main`, weekly, and on demand. It is the gate; everything in it should stay
bounded in runtime.

| Job | What it does |
| --- | --- |
| Typos | `crate-ci/typos` over the workflows, `volumential/`, `README.md`, `DEVELOPMENT.md` and `pyproject.toml`, configured by `.typos.toml` |
| Ruff | `ruff check --select E9,F63,F7,F82` — the error-level smoke subset, not the full `ruff.toml` rule set |
| Type checking | `basedpyright -p pyproject.toml --level error` |
| Testing (Linux) | the default pytest suite under a micromamba environment, with a wrapper timeout and a diagnostics artifact (`linux-pytest.log`, `pytest.xml`) uploaded on every outcome |
| Examples (Smoke) | the maintained examples under `VOLUMENTIAL_EXAMPLE_SMOKE=1`, plus several benchmark drivers in `--mode smoke` |

`PYOPENCL_CTX` and `PYOPENCL_TEST` are pinned to `portable:0` at the workflow
level, so CI always runs on PoCL rather than on whatever enumerates first.

## `CI Full` — `main`, weekly, and on demand

`.github/workflows/ci-full.yml` does not run on pull requests.

| Job | What it does |
| --- | --- |
| Testing (macOS) | the suite on macOS |
| Documentation | `sphinx-build` of this site; the job installs `.[test,doc]` |
| Full Accuracy Tests | scheduled or manual only — collects **and** runs the `full_accuracy` marker |
| Examples | the maintained examples at full settings, with a cached Laplace 3D table |

The full-accuracy job runs on GitHub-hosted CPU runners. Full numerical
execution of the marked tests still wants a GPU-capable environment; the value
of running the collection step explicitly is that marker drift, import errors
and accidental deselection surface instead of being mistaken for validation.

## Adding a documentation dependency

Every new documentation dependency goes into the `doc` extra of
`pyproject.toml`. The `Documentation` job of `CI Full` installs `.[test,doc]`,
so anything in the base dependencies, the `test` extra or the `doc` extra is
installed for it; only an extra that job does not select (`benchmark`,
`fmmlib`, `gmsh_support`) is missing. The `doc` extra is where a documentation
dependency belongs regardless, because it is what
`uv sync --active --extra doc` gives a contributor locally.

Note where the failure would surface: `CI Full` has no `pull_request` trigger,
so a documentation dependency that is not installed fails on `main`, not on the
pull request that introduced it.

## The review bots

Pull requests are reviewed by the maintainer and by two automated services:

- **OpenAI Codex** reviews when a pull request is marked ready for review.
  Comment `@codex review` on the pull request to request another pass after
  pushing fixes.
- **CodeRabbit** reviews on the same trigger and comments inline.

Treat their findings as review comments, not as a gate: fix the correct ones,
reply on the thread saying what changed, and resolve it. The maintainer merges;
the bots do not.

Because the bots review on *ready for review*, a draft pull request gets no
automated pass. Open it ready when you want one.

## Stacked pull requests

A branch stacked on another open pull request targets that branch, not `main`,
and says so in its body. Merge the stack in order, and **do not delete a base
branch while a child is still open** — deleting a base closes its stacked
children.
