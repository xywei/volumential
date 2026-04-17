import numpy as np


def pick_far_positive_case_id(table):
    case_vecs = np.asarray(table.interaction_case_vecs, dtype=np.int64)
    positive_ids = [i for i, vec in enumerate(case_vecs) if np.all(vec > 0)]
    if not positive_ids:
        positive_ids = list(range(len(case_vecs)))
    return max(positive_ids, key=lambda i: int(np.dot(case_vecs[i], case_vecs[i])))
