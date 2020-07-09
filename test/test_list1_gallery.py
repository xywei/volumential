from __future__ import absolute_import, division, print_function

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

import volumential.list1_gallery as l1g


def test_1d():
    l1g1d, _, _ = l1g.generate_list1_gallery(1)
    assert len(l1g1d) == 6 + 1

    distinct_numbers = set()
    for vec in l1g1d:
        for cvc in vec:
            distinct_numbers.add(cvc)

    assert len(distinct_numbers) == 7
    assert max(distinct_numbers) == 6
    assert min(distinct_numbers) == -6


def test_2d():
    l1g2d, _, _ = l1g.generate_list1_gallery(2)
    assert len(l1g2d) == 8 + 12 + 12 + 1

    distinct_numbers = set()
    for vec in l1g2d:
        for cvc in vec:
            distinct_numbers.add(cvc)

    assert len(distinct_numbers) == 11
    assert max(distinct_numbers) == 6
    assert min(distinct_numbers) == -6


def test_3d():
    l1g3d, _, _ = l1g.generate_list1_gallery(3)

    distinct_numbers = set()
    for vec in l1g3d:
        for cvc in vec:
            distinct_numbers.add(cvc)

    assert max(distinct_numbers) == 6
    assert min(distinct_numbers) == -6
