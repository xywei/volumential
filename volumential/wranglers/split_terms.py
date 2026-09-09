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

__doc__ = """Bookkeeping for Helmholtz-split term tables.

Owns the canonical form of a split term key, the rho-driven choice of split
order, and the cache-accounting record reported by the wrangler.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HelmholtzSplitCacheAccounting:
    split_enabled: bool
    split_order: int
    parameter_count: int
    base_table_count: int
    split_term_table_count: int
    basis_table_count: int
    base_table_payload_bytes: int
    split_term_table_payload_bytes: int
    total_table_payload_bytes: int
    split_term_keys: tuple
    uses_online_coefficients: bool
    uses_online_remainder: bool


def _nearfield_table_payload_bytes(table):
    if bool(getattr(table, "table_data_is_symmetry_reduced", False)):
        if hasattr(table, "get_reduced_table_data"):
            _, values = table.get_reduced_table_data()
            return int(np.asarray(values).nbytes)
        data = np.asarray(table.data)
        return int(np.count_nonzero(np.isfinite(data)) * data.dtype.itemsize)
    data = np.asarray(table.data)
    return int(data.nbytes)


def _normalize_helmholtz_split_term_key(term_key):
    if isinstance(term_key, tuple):
        if len(term_key) != 2:
            raise ValueError(
                "helmholtz split term key tuples must have length 2: (kind, power)"
            )
        kind, power = term_key
    elif isinstance(term_key, int):
        kind = "power"
        power = term_key
    elif isinstance(term_key, str):
        if term_key == "constant":
            kind = "power"
            power = 0
        elif term_key.startswith("power:"):
            kind = "power"
            power = term_key.split(":", 1)[1]
        elif term_key.startswith("power_log:"):
            kind = "power_log"
            power = term_key.split(":", 1)[1]
        else:
            raise ValueError(
                "invalid helmholtz split term key string; expected one of: "
                "constant, power:<n>, power_log:<n>"
            )
    else:
        raise TypeError(
            "helmholtz split term key must be int, str, or (kind, power) tuple"
        )

    kind = str(kind)
    if kind not in ("power", "power_log"):
        raise ValueError("helmholtz split term key kind must be 'power' or 'power_log'")

    ipower = int(power)
    if ipower < 0:
        raise ValueError("helmholtz split term key power must be >= 0")
    if kind == "power_log" and ipower == 0:
        raise ValueError("helmholtz split power_log term requires power >= 1")

    return (kind, ipower)


def _format_helmholtz_split_term_key(term_key):
    kind, power = _normalize_helmholtz_split_term_key(term_key)
    if kind == "power":
        return f"power:{power}"
    return f"power_log:{power}"


def _select_split_order_from_rho(rho_max, thresholds, orders):
    if len(orders) != len(thresholds) + 1:
        raise ValueError("orders must have one more element than thresholds")

    rho = float(rho_max)
    for idx, threshold in enumerate(thresholds):
        if rho <= float(threshold):
            return int(orders[idx])
    return int(orders[-1])


def _select_split_order_from_rho_components(
    rho_real,
    rho_imag,
    thresholds_real,
    thresholds_imag,
    orders,
):
    order_real = _select_split_order_from_rho(rho_real, thresholds_real, orders)
    order_imag = _select_split_order_from_rho(rho_imag, thresholds_imag, orders)
    return max(order_real, order_imag)

# vim: filetype=pyopencl:foldmethod=marker
