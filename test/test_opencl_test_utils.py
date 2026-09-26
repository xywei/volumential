"""Device selection of :func:`_opencl_test_utils.create_fp64_context_or_skip`.

The platforms, devices and contexts here are stand-ins, so these tests pin the
selection rule on any host -- including the branch that prefers a GPU, which a
CPU-only runner could not otherwise exercise.
"""

from types import SimpleNamespace

import pytest

import pyopencl as cl


try:
    import _opencl_test_utils as utils
except ImportError:
    import test._opencl_test_utils as utils


GPU = cl.device_type.GPU
CPU = cl.device_type.CPU


def _device(name, device_type, *, fp64=True):
    return SimpleNamespace(
        name=name,
        type=device_type,
        extensions="cl_khr_fp64" if fp64 else "",
        double_fp_config=0,
    )


def _platform(name, *devices):
    return SimpleNamespace(name=name, get_devices=lambda: list(devices))


@pytest.fixture
def fake_cl(monkeypatch):
    """Replace platform discovery and context creation with stand-ins.

    Returns a namespace whose ``platforms`` the test fills in, and which
    records the ``answers`` passed to ``create_some_context``.
    """
    state = SimpleNamespace(platforms=[], answers=None)

    def create_some_context(interactive=None, answers=None):
        assert interactive is False
        state.answers = answers
        platform_name, device_index = answers
        for platform in state.platforms:
            if platform_name.lower() in platform.name.lower():
                return SimpleNamespace(
                    devices=[platform.get_devices()[int(device_index)]]
                )
        raise RuntimeError("input did not match any platform")

    monkeypatch.delenv("PYOPENCL_CTX", raising=False)
    monkeypatch.setattr(cl, "get_platforms", lambda: state.platforms)
    monkeypatch.setattr(cl, "create_some_context", create_some_context)
    monkeypatch.setattr(
        cl, "Context", lambda devices: SimpleNamespace(devices=devices)
    )
    return state


def test_default_prefers_an_fp64_gpu(fake_cl):
    gpu = _device("gpu", GPU)
    fake_cl.platforms = [
        _platform("Portable Computing Language", _device("cpu", CPU)),
        _platform("NVIDIA CUDA", _device("gpu without fp64", GPU, fp64=False), gpu),
    ]
    assert utils.create_fp64_context_or_skip().devices == [gpu]


def test_default_falls_back_to_an_fp64_cpu(fake_cl):
    cpu = _device("cpu", CPU)
    fake_cl.platforms = [
        _platform("NVIDIA CUDA", _device("gpu without fp64", GPU, fp64=False)),
        _platform("Portable Computing Language", cpu),
    ]
    assert utils.create_fp64_context_or_skip().devices == [cpu]


def test_default_never_picks_the_intel_cpu_runtime(fake_cl):
    fake_cl.platforms = [
        _platform(utils.INTEL_OPENCL_PLATFORM_NAME, _device("intel cpu", CPU)),
    ]
    with pytest.raises(pytest.skip.Exception, match="fp64"):
        utils.create_fp64_context_or_skip()


def test_pyopencl_ctx_selects_a_cpu_over_an_available_gpu(fake_cl, monkeypatch):
    cpu = _device("cpu", CPU)
    fake_cl.platforms = [
        _platform("NVIDIA CUDA", _device("gpu", GPU)),
        _platform("Portable Computing Language", cpu),
    ]
    monkeypatch.setenv("PYOPENCL_CTX", "portable:0")
    monkeypatch.setenv("PYOPENCL_TEST", "cuda:0")

    assert utils.create_fp64_context_or_skip().devices == [cpu]
    assert fake_cl.answers == ["portable", "0"]


def test_pyopencl_ctx_may_select_the_intel_cpu_runtime(fake_cl, monkeypatch):
    intel_cpu = _device("intel cpu", CPU)
    fake_cl.platforms = [_platform(utils.INTEL_OPENCL_PLATFORM_NAME, intel_cpu)]
    monkeypatch.setenv("PYOPENCL_CTX", "intel:0")

    assert utils.create_fp64_context_or_skip().devices == [intel_cpu]


def test_pyopencl_ctx_without_fp64_skips_instead_of_substituting(
    fake_cl, monkeypatch
):
    fake_cl.platforms = [
        _platform("NVIDIA CUDA", _device("gpu", GPU)),
        _platform("Portable Computing Language", _device("cpu", CPU, fp64=False)),
    ]
    monkeypatch.setenv("PYOPENCL_CTX", "portable:0")

    with pytest.raises(pytest.skip.Exception, match="without fp64"):
        utils.create_fp64_context_or_skip()


def test_pyopencl_ctx_matching_nothing_is_an_error(fake_cl, monkeypatch):
    fake_cl.platforms = [
        _platform("Portable Computing Language", _device("cpu", CPU)),
    ]
    monkeypatch.setenv("PYOPENCL_CTX", "cuda:0")

    with pytest.raises(RuntimeError, match="did not match"):
        utils.create_fp64_context_or_skip()
