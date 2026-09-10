Development
===========

The canonical development environment guide is maintained in
``DEVELOPMENT.md`` at the repository root.

It covers:

- local setup with ``uv`` and OpenCL runtime prerequisites,
- the dependency-provisioning rules that promoted evidence depends on
  (inducer stack from Git sources, ``pyfmmlib`` with OpenMP and the batched
  P2M wrappers, the post-provisioning traversal sanity check, and the thread
  caps to record in run metadata),
- remote setup for heavier numerical experiments,
- lint, type-check and test commands.

Lint configuration lives in ``ruff.toml`` at the repository root: 85 columns,
Python 3.11 target, and a ``[lint.per-file-ignores]`` baseline of pre-existing
violations that shrinks as files are cleaned.

Use this alongside :doc:`install` for day-to-day development workflows.
