import volumential.version as version


def test_fallback_kernel_revision_is_stable():
    first = version._fallback_kernel_revision()
    second = version._fallback_kernel_revision()

    assert first == second
    assert first.startswith("nogit-")
    assert len(first) > len("nogit-")
