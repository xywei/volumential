from __future__ import absolute_import, division, print_function


def test_import():
    try:
        # Silence flake8 on this import
        import volumential  # NOQA
        assert True
    except ImportError:
        assert False
