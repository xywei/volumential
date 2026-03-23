import volumential.version as version


def test_fallback_kernel_revision_is_stable():
    first = version._fallback_kernel_revision()
    second = version._fallback_kernel_revision()

    assert first == second
    assert first.startswith("nogit-")
    assert len(first) > len("nogit-")


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
