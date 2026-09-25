# CI and the review bots

Three GitHub Actions workflows and two automated reviewers stand between a
branch and a published site: `CI` gates a pull request, `CI Full` carries the
long jobs on `main`, and `Docs Pages` publishes this site.

## `CI` — pull requests targeting `main`

`.github/workflows/ci.yml` runs on pull requests **whose base is `main`**, on
pushes to `main`, weekly, and on demand. It is the gate; everything in it
should stay bounded in runtime.

The base-branch restriction has a consequence worth stating before the table:
a stacked pull request, which targets its parent branch rather than `main`,
gets **none** of these checks. See
[Stacked pull requests](#stacked-pull-requests).

| Job | What it does |
| --- | --- |
| Typos | `crate-ci/typos` over the workflows, `volumential/`, `README.md`, `DEVELOPMENT.md` and `pyproject.toml`, configured by `.typos.toml` |
| Ruff | `ruff check --select E9,F63,F7,F82` — the error-level smoke subset, not the full `ruff.toml` rule set |
| Type checking | `basedpyright -p pyproject.toml --level error` |
| Testing (Linux) | the default pytest suite under a micromamba environment, installing `.[test]` plus the `pyfmmlib` commit that `uv.lock` pins (the `fmmlib` extra's content, passed as a pinned requirement so uv sees one Git URL), with a wrapper timeout and a diagnostics artifact (`linux-pytest.log`, `pytest.xml`) uploaded on every outcome |
| Examples (Smoke) | four examples under `VOLUMENTIAL_EXAMPLE_SMOKE=1` — `laplace2d.py`, `laplace2d_adaptive.py`, `helmholtz2d.py`, `helmholtz3d.py` — under `set -euo pipefail`; the two Laplace examples run through `doc/tools/render_gallery.py`, and the upload of their figures and manifest as a `gallery-laplace2d-*` artifact is attempted on every outcome, so the artifact exists whenever the renderer wrote files, even if a later example fails (see {doc}`gallery-assets`) |
| Documentation | this site: `sphinx-build -W --keep-going -n -b html`, then the two coverage reports (`-b coverage` and `interrogate`), then `-b linkcheck` last. The built HTML is uploaded as a `docs-html-*` artifact and the reports as `docs-coverage-*`; the job installs `.[test,doc]` |

`PYOPENCL_CTX` and `PYOPENCL_TEST` are pinned to `portable:0` at the workflow
level, so CI always runs on PoCL rather than on whatever enumerates first.
The documentation job needs no device, but it does import `volumential`, so it
uses the same micromamba environment as the rest: `pyopencl` and `loopy` have
to be importable.

Two details of these jobs are load-bearing rather than incidental, and both
were added in [#151](https://github.com/xywei/volumential/issues/151):

- The smoke job runs every one of its commands in **one** `run:` block, and
  the shell `setup-micromamba` generates does not pass `-e`. Without the
  `set -euo pipefail` that now opens the block, only the last command's exit status
  reaches GitHub, and a failing example is reported green — which is exactly
  what happened to `helmholtz2d.py` for as long as it was broken. Keep the
  `set` line if the block is ever rewritten, or give each command its own step.
- `Testing (Linux)` installs the `fmmlib` extra, which resolves `pyfmmlib` to
  a Git source rather than to a release and so **builds it from source**
  against the Fortran compiler `.test-conda-env-py3.yml` installs. That build
  is what gives `test/test_fmmlib_batched_stages.py` its batched
  `{l,h}{2,3}dformmp_imany` entry points; against a released wheel, PyPI's or
  conda-forge's, eight of its tests skip and the batched-P2M path has no CI
  coverage (see {doc}`../getting-started/installation`).

  Two details of *that* are worth knowing before editing the step. First,
  `uv pip install` does not read `uv.lock` — only `uv sync` does, and an exact
  sync would uninstall the conda-provided half of the environment — so the
  `[tool.uv.sources]` entry, which names a repository but no revision, would
  float on upstream `main`. `.github/scripts/pyfmmlib_requirement.py` reads the
  locked commit and prints it as a requirement, which the step passes to `uv`;
  `uv.lock` therefore stays both the one place the revision is written down and
  the thing that decides what CI builds. Second, the step then imports the four
  wrappers, because `FPNDFMMLibExpansionWrangler` falls back to the serial
  per-box path *without complaining* when they are missing: a build that
  silently fell back has to fail the environment rather than quietly reduce the
  suite to what it covered before.

The link check runs last on purpose: it is the only step whose outcome depends
on hosts nobody here controls, so a rate-limited or unreachable third party
cannot stop the deterministic checks from being reported. The URLs that cannot
be checked from a runner at all — a publisher and an OpenCL specification page
that answer a runner with 403, and the local `sphinx-autobuild` address — are
named one by one in `linkcheck_ignore` in `conf.py`, each with its reason. They
are spelled out as single URLs rather than as hosts, so that the next link to
the same site is still checked.

To review a change to the site rather than to its build, download the
`docs-html-*` artifact from the run's **Artifacts** section and open
`index.html`. It is built with the command the deployment uses, so it is what
would go live; it is kept for 14 days.

Note what the smoke job does **not** cover: `laplace3d.py`, `poisson3d.py`
and `branched_flow_helmholtz2d.py` run only in the `Examples` job of
`CI Full`, which has no `pull_request` trigger. A change that breaks one of
those is not caught before it reaches `main`.

## `CI Full` — `main`, weekly, and on demand

`.github/workflows/ci-full.yml` does not run on pull requests.

The documentation build used to live here. It moved into `CI` in 2026-09, so
that a broken page, a dead link or a docstring that fell off the API reference
fails on the pull request that caused it rather than on `main`; `CI` also runs
on pushes to `main` and on the weekly schedule, so nothing was given up by the
move.

| Job | What it does |
| --- | --- |
| Testing (macOS) | the suite on macOS |
| Full Accuracy Tests | scheduled or manual only — collects **and** runs the `full_accuracy` marker |
| Examples | the maintained examples at full settings, with a cached Laplace 3D table |

The full-accuracy job runs on GitHub-hosted CPU runners. Full numerical
execution of the marked tests still wants a GPU-capable environment; the value
of running the collection step explicitly is that marker drift, import errors
and accidental deselection surface instead of being mistaken for validation.

## Adding a documentation dependency

Every new documentation dependency goes into the `doc` extra of
`pyproject.toml`. Both jobs that build this site — `Documentation` in `CI` and
`Build` in `Docs Pages` — install `.[test,doc]`, so anything in the base
dependencies, the `test` extra or the `doc` extra is installed for them; only
an extra that neither selects (`fmmlib` and `gmsh_support`, for instance) is
missing.
The `doc` extra is where a documentation dependency belongs regardless, because
it is what `uv sync --extra test --extra doc` gives a contributor locally.

A dependency left out of the extra fails the `Documentation` job of the pull
request that introduced it, which is the point of the job running there.

## `Docs Pages` — publishing the site

`.github/workflows/docs-pages.yml` builds this site from `main` and deploys it
to GitHub Pages. It runs on every push to `main`, and manually — a
`workflow_dispatch` exists for the first publication, when there may be no new
commit to trigger one. Both jobs are guarded by
`if: github.ref == 'refs/heads/main'`, because a dispatch takes a ref and an
unguarded one would publish a feature branch to the production site.

It has two jobs: `Build`, which runs the same
`sphinx-build -W --keep-going -n -b html` as `CI` and hands the output to
`actions/upload-pages-artifact`, and `Deploy`, which calls
`actions/deploy-pages`. The workflow is read-only by default and the two
grants a deployment needs — `pages: write`, and `id-token: write` for the OIDC
token the deploy action exchanges for one — sit on `Deploy` alone, so the job
that runs the build cannot reach the Pages API. It serialises on a single
`pages` concurrency group with
`cancel-in-progress: false`, so two pushes queue rather than race and a deploy
in flight is never cancelled half-way.

The link check and the coverage reports are deliberately *not* repeated here.
They gate a change in `CI`; a third-party host going down afterwards should not
block the deployment of pages that already passed them.

Pages has been enabled for the repository since 2026-09-13 (**Settings → Pages
→ Build and deployment → Source: `GitHub Actions`**, a setting the maintainer
owns), and the site is <https://xywei.github.io/volumential/>. The workflow
does not enable Pages for itself — `actions/configure-pages` can, with
`enablement: true` and a token beyond `GITHUB_TOKEN`, and turning on a public
site is not a decision a workflow should make. If the setting is ever turned
off again, `Build` still succeeds and `Deploy` fails with *Get Pages site
failed*; nothing else is affected.

With the site live, the version switcher's `json_url` in `doc/source/conf.py`
is the absolute `https://xywei.github.io/volumential/_static/switcher.json`
(so a future tagged build reads the current index rather than its own frozen
copy), `README.md` and the repository homepage point at the site, and
`html_baseurl` names the same URL for the `canonical` links, `sitemap.xml` and
the OpenGraph metadata.

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

Two things do not work on a stacked pull request, and both are worth planning
around:

- **`CI` does not run.** Its `pull_request` trigger is restricted to base
  `main`, so a stacked pull request has no lint, type, test or example job.
  Run them locally, and expect the first check to happen when the stack
  reaches `main`.
- **CodeRabbit does not auto-review**, because auto-reviews are limited to the
  default branch. Ask for one with `@coderabbitai review`. Codex reviews
  stacked pull requests normally.
