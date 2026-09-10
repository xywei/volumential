"""Tests for the version and kernel-revision helpers."""

from volumential import version


def test_fallback_kernel_revision_is_stable():
    first = version._fallback_kernel_revision()
    second = version._fallback_kernel_revision()

    assert first == second
    assert first.startswith("nogit-")
    assert len(first) > len("nogit-")


def test_fallback_kernel_revision_covers_every_generated_kernel_module():
    """Every module that defines a loopy kernel must move the token.

    Without a git revision, ``KERNEL_VERSION`` is this fingerprint, and it is
    the cache key ``KernelCacheWrapper`` reuses compiled kernels under. A
    module that defines generated-kernel code but is not hashed would let an
    upgrade serve a stale binary under an unchanged key. ``tools.py`` and
    ``expansion_wrangler_fpnd.py`` are now re-export shims, so hashing them
    is not enough on its own.
    """
    import shutil
    import tempfile
    from pathlib import Path

    source_root = Path(version.__file__).resolve().parent

    generated_kernel_modules = [
        "kernel_cache.py",
        "expression_eval.py",
        "box_operators.py",
        "volume_fmm.py",
        "list1.py",
        "nearfield_potential_table.py",
    ]
    generated_kernel_modules += sorted(
        f"wranglers/{path.name}"
        for path in (source_root / "wranglers").glob("*.py")
    )

    baseline = version._fallback_kernel_revision()

    with tempfile.TemporaryDirectory() as scratch:
        copy_root = Path(scratch) / "volumential"
        shutil.copytree(source_root, copy_root)

        # the helper reads relative to the module file, so point it at a copy
        original_file = version.__file__
        try:
            version.__file__ = str(copy_root / "version.py")
            assert version._fallback_kernel_revision() == baseline

            for rel_path in generated_kernel_modules:
                target = copy_root / rel_path
                original = target.read_bytes()
                target.write_bytes(original + b"\n# perturbation\n")
                try:
                    assert version._fallback_kernel_revision() != baseline, (
                        f"{rel_path} does not move the fallback revision"
                    )
                finally:
                    target.write_bytes(original)

            assert version._fallback_kernel_revision() == baseline
        finally:
            version.__file__ = original_file


def test_resolve_git_revision_ignores_stale_generated_module(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytools

    monkeypatch.setitem(
        sys.modules,
        "volumential._git_rev",
        SimpleNamespace(GIT_REVISION="stale-generated-revision"),
    )
    monkeypatch.setattr(
        pytools,
        "find_module_git_revision",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        version,
        "_fallback_kernel_revision",
        lambda: "nogit-test-fallback",
    )

    assert version._resolve_git_revision() == "nogit-test-fallback"
