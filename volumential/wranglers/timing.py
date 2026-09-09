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

__doc__ = """Timing-future adapter for the boxtree FMM driver.
"""

class SumpyTimingFuture:
    """The timing handle boxtree's FMM driver expects from a wrangler stage.

    Calling it waits on the stage's OpenCL events.  The elapsed time itself is
    not collected -- the wranglers do their profiling elsewhere -- so the call
    reports ``0.0``.
    """

    def __init__(self, queue, events) -> None:
        self.queue = queue
        self.events = [evt for evt in events if evt is not None]

    def __call__(self) -> float:
        for evt in self.events:
            evt.wait()
        return 0.0

# vim: filetype=pyopencl:foldmethod=marker
