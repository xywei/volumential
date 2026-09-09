"""Helpers shared by the Duffy-radial table tests."""

import numpy as np


def pick_far_positive_case_id(table) -> int:
    """Return the id of the farthest strictly-positive interaction case.

    Falls back to the farthest case of any sign when the table has no case
    whose displacement vector is positive in every axis.
    """
    case_vecs = np.asarray(table.interaction_case_vecs, dtype=np.int64)
    positive_ids = [i for i, vec in enumerate(case_vecs) if np.all(vec > 0)]
    if not positive_ids:
        positive_ids = list(range(len(case_vecs)))
    return max(positive_ids, key=lambda i: int(np.dot(case_vecs[i], case_vecs[i])))
