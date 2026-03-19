import numpy as np


class _FakeContext:
    def __init__(self, int_ptr):
        self.int_ptr = int_ptr


class _FakeQueue:
    def __init__(self, context):
        self.context = context


class _FakeProgram:
    def __init__(self, counter):
        self._counter = counter

    def executor(self, _context):
        self._counter["executor"] += 1

        def _executor(*_args, **_kwargs):
            return None, {"result": None}

        return _executor


def test_list1_executor_cache():
    import volumential.list1 as list1

    list1._clear_list1_executor_cache()

    counter = {"executor": 0}
    program = _FakeProgram(counter)
    queue = _FakeQueue(_FakeContext(101))

    ex0 = list1._get_list1_executor(queue, program, ("kernel", 1))
    ex1 = list1._get_list1_executor(queue, program, ("kernel", 1))
    ex2 = list1._get_list1_executor(queue, program, ("kernel", 2))

    assert ex0 is ex1
    assert ex0 is not ex2
    assert counter["executor"] == 2


def test_nearfield_fused_duffy_executor_cache(monkeypatch):
    from volumential.nearfield_potential_table import NearFieldInteractionTable

    table = NearFieldInteractionTable.__new__(NearFieldInteractionTable)
    table.quad_order = 4
    table.dim = 2
    table.dtype = np.float64
    table._fused_duffy_executor_cache = {}

    counter = {"program": 0, "executor": 0}

    def fake_program(_queue, _n_entries, _n_nodes):
        counter["program"] += 1
        return _FakeProgram(counter)

    monkeypatch.setattr(table, "_get_fused_invariant_duffy_table_program", fake_program)

    queue = _FakeQueue(_FakeContext(303))

    ex0 = table._get_fused_invariant_duffy_table_executor(queue, 8, 16)
    ex1 = table._get_fused_invariant_duffy_table_executor(queue, 8, 16)
    ex2 = table._get_fused_invariant_duffy_table_executor(queue, 8, 32)

    assert ex0 is ex1
    assert ex0 is not ex2
    assert counter["program"] == 2
    assert counter["executor"] == 2


def test_nearfield_fused_duffy_executor_no_context(monkeypatch):
    from volumential.nearfield_potential_table import NearFieldInteractionTable

    table = NearFieldInteractionTable.__new__(NearFieldInteractionTable)
    table.quad_order = 4
    table.dim = 2
    table.dtype = np.float64
    table._fused_duffy_executor_cache = {}

    counter = {"program": 0}

    program = object()

    def fake_program(_queue, _n_entries, _n_nodes):
        counter["program"] += 1
        return program

    monkeypatch.setattr(table, "_get_fused_invariant_duffy_table_program", fake_program)

    class QueueNoContext:
        pass

    queue = QueueNoContext()
    got = table._get_fused_invariant_duffy_table_executor(queue, 8, 16)

    assert got is program
    assert counter["program"] == 1
