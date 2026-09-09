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

__doc__ = """Open-addressed integer lookup tables for orbit reconstruction.

The generated reconstruction maps are consumed by OpenCL kernels, so they are
materialised as flat probe-sequence arrays rather than Python dicts.
"""

import math

import numpy as np


_ORBIT_RECONSTRUCTION_HASH_MULTIPLIER = 33


def _build_open_addressed_int_lookup(
    keys, *, max_load_factor=0.5, target_max_probe_count=16
):
    keys = np.asarray(keys, dtype=np.int64)
    if keys.ndim != 1:
        raise ValueError("lookup keys must be one-dimensional")
    if len(keys) == 0:
        raise ValueError("lookup keys cannot be empty")
    if np.any(keys < 0) or np.any(keys > np.iinfo(np.int32).max):
        raise ValueError("lookup keys must fit in nonnegative int32")

    size = 1
    min_size = int(math.ceil(len(keys) / float(max_load_factor)))
    while size < min_size:
        size *= 2

    while True:
        lookup_keys = np.full(size, -1, dtype=np.int32)
        lookup_values = np.full(size, -1, dtype=np.int32)
        max_probe_count = 0
        mask = size - 1

        for value, key64 in enumerate(keys.tolist()):
            key = int(key64)
            slot = (key * _ORBIT_RECONSTRUCTION_HASH_MULTIPLIER) & mask
            probe_count = 1
            while lookup_keys[slot] != -1:
                if int(lookup_keys[slot]) == key:
                    raise ValueError("lookup keys must be unique")
                slot = (slot + 1) & mask
                probe_count += 1
            lookup_keys[slot] = key
            lookup_values[slot] = value
            max_probe_count = max(max_probe_count, probe_count)

        if max_probe_count <= target_max_probe_count or size >= 64 * len(keys):
            return lookup_keys, lookup_values, max_probe_count

        size *= 2


def _build_sparse_sign_lookup(keys, values):
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.int8)
    if keys.shape != values.shape:
        raise ValueError("sign lookup keys and values must have the same shape")

    if len(keys) == 0:
        return (
            np.full(1, -1, dtype=np.int32),
            np.ones(1, dtype=np.int8),
            1,
        )

    lookup_keys, lookup_values_i32, max_probe_count = _build_open_addressed_int_lookup(
        keys,
        max_load_factor=0.5,
    )
    lookup_values = np.ones(lookup_values_i32.shape, dtype=np.int8)
    occupied = lookup_values_i32 >= 0
    lookup_values[occupied] = values[lookup_values_i32[occupied]]
    return lookup_keys, lookup_values, max_probe_count

# vim: filetype=pyopencl:foldmethod=marker
