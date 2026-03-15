import types

import pytest

import volumential.function_extension as fe


def test_setup_cl_ctx_prefers_queue_context():
    queue = types.SimpleNamespace(context="ctx-from-queue")
    assert fe.setup_cl_ctx(queue=queue) == "ctx-from-queue"


def test_setup_cl_ctx_prefers_explicit_context():
    queue = types.SimpleNamespace(context="ctx-from-queue")
    assert fe.setup_cl_ctx(ctx="explicit-ctx", queue=queue) == "explicit-ctx"


def test_setup_command_queue_returns_existing_queue():
    queue = object()
    assert fe.setup_command_queue(queue=queue) is queue


def test_get_tangent_vectors_rotates_normals(monkeypatch):
    normal = [2, -3]

    monkeypatch.setattr(fe, "get_normal_vectors", lambda queue, discr, loc_sign: normal)

    tangent = fe.get_tangent_vectors(queue=None, density_discr=None, loc_sign=1)

    assert tangent[0] == 3
    assert tangent[1] == 2


@pytest.mark.parametrize("loc_sign, bound_value, expected", [(1, 1, -1), (-1, 1, 1)])
def test_get_normal_vectors_uses_expected_orientation(
    monkeypatch, loc_sign, bound_value, expected
):
    calls = []

    class FakeNormal:
        def __init__(self, dim):
            self.dim = dim

        def as_vector(self):
            return ("normal", self.dim)

    monkeypatch.setattr(fe.sym, "normal", lambda dim: FakeNormal(dim))

    def fake_bind(discr, expr):
        calls.append((discr, expr))

        def evaluator(queue):
            return bound_value

        return evaluator

    monkeypatch.setattr(fe, "bind", fake_bind)

    result = fe.get_normal_vectors(
        queue="queue", density_discr="discr", loc_sign=loc_sign
    )

    assert result == expected
    assert calls == [("discr", ("normal", 2))]
