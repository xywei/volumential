__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# {{{ find install- or run-time git revision

import os
from hashlib import blake2b
from pathlib import Path


def _fallback_kernel_revision():
    """Return a deterministic fingerprint for cache invalidation.

    Built artifacts installed outside of a git checkout do not carry a runtime
    revision from :func:`pytools.find_module_git_revision`. When that happens,
    derive a revision-like token from a stable subset of source files so cached
    kernels do not silently reuse stale binaries across upgrades.
    """
    source_root = Path(__file__).resolve().parent
    fingerprint = blake2b(digest_size=12)

    for rel_path in (
        "version.py",
        "tools.py",
        "volume_fmm.py",
        "list1.py",
        "nearfield_potential_table.py",
        "expansion_wrangler_fpnd.py",
    ):
        try:
            payload = (source_root / rel_path).read_bytes()
        except OSError:
            continue

        fingerprint.update(rel_path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(payload)

    return f"nogit-{fingerprint.hexdigest()}"


if os.environ.get("AKPYTHON_EXEC_FROM_WITHIN_WITHIN_SETUP_PY") is not None:
    # We're just being exec'd by setup.py. We can't import anything.
    _git_rev = None

else:
    try:
        import volumential._git_rev as _git_rev_mod
    except ImportError:
        _git_rev = None
    else:
        _git_rev = _git_rev_mod.GIT_REVISION

    # If we're running from a dev tree, the last install (and hence the most
    # recent update of the above git rev) could have taken place very long ago.
    from pytools import find_module_git_revision

    _runtime_git_rev = find_module_git_revision(__file__, n_levels_up=1)
    if _runtime_git_rev is not None:
        _git_rev = _runtime_git_rev
    elif _git_rev is None:
        _git_rev = _fallback_kernel_revision()

# }}}

VERSION = (2017, 1)
VERSION_STATUS = "a0"
VERSION_TEXT = ".".join(str(i) for i in VERSION) + VERSION_STATUS

KERNEL_VERSION = (VERSION, _git_rev, 0)

LOOPY_LANG_VERSION = (2018, 2)

# vim: ft=pyopencl
