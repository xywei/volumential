"""Per-phase wall-clock profiling for the Paper 1 benchmark drivers (E6).

The operation-count cost model prices *provisioning* strategies (see
:mod:`volumential.opcounters`).  Experiment E6 asks the complementary
question about an *end-to-end solve*: which phase of one solve owns which
share of the work, in operations and in seconds.  This module supplies the
seconds half; :func:`volumential.opcounters.fmm_stage_operation_counts`
supplies the operations half, phase for phase.

The phases of :func:`volumential.volume_fmm.drive_volume_fmm` are the FMM
stage graph plus the two near-field phases:

===============================  ===================================
phase name                       stage
===============================  ===================================
``far_form_multipoles``          P2M
``far_coarsen_multipoles``       M2M (upward pass)
``nearfield_table_apply``        List 1 through the base near-field table
``split_correction``             online split correction: series remainder
                                 P2P, the retained-channel table applies,
                                 and the smooth-source rebuild they need
``far_multipole_to_local``       M2L (List 2)
``far_eval_multipoles``          M2P (List 3 far)
``far_form_locals``              P2L (List 4 far)
``far_refine_locals``            L2L (downward pass)
``far_eval_locals``              L2P
===============================  ===================================

Design constraints, in order of importance:

1. **Zero cost when inactive.**  :func:`phase` is a plain function that
   short-circuits on one :class:`~contextvars.ContextVar` read and returns
   a shared do-nothing context manager when no profile is active, so the
   instrumented driver keeps its uninstrumented timings.  Deliberately not
   a ``@contextmanager`` generator: building and entering one costs about
   1.4 us, which ``drive_volume_fmm`` would pay eight or nine times per
   solve -- eight versus nine, because only the online-split path enters
   ``split_correction``, so the "inert" instrumentation would put a
   strategy-dependent bias into exactly the unprofiled solve means the
   break-even curves are built from.  Nothing is monkeypatched and no timer
   runs unless a caller opted in.

2. **Activation follows the caller, not the process.**  The active profiles
   live in a :class:`~contextvars.ContextVar`, so two threads that each
   profile a solve record only their own phases and drain only their own
   OpenCL queue.  A process-global activation would silently merge them.

3. **Device work is attributed to the phase that launched it.**  OpenCL
   command queues are asynchronous, so a host-side timer around a kernel
   launch measures the launch, not the kernel.  A profile therefore carries
   a *sync* callable (in practice ``queue.finish``) which :func:`phase`
   invokes on entry and again before stopping the clock.  This serializes
   the queue at every phase boundary, so a profiled solve is *not* a
   faithful measurement of an unprofiled solve's total wall time: drivers
   must run profiled solves separately from the solves whose totals they
   report, and must report the profiled total alongside the shares so the
   perturbation is visible.

4. **Phases are disjoint by construction, not by assumption.**  The blocks
   in ``drive_volume_fmm`` do not nest.  If a caller nests them anyway, the
   inner name is recorded in :attr:`PhaseProfile.nested_names` and the
   elapsed time is counted under both names; a consumer that finds
   ``nested_names`` non-empty must not read the shares as a partition.

Whatever the profiled phases do not cover -- reordering sources and
potentials, finalization, host bookkeeping -- is the caller's business to
report as a residual (profiled solve total minus the sum of the phases).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "FAR_FIELD_PHASES",
    "NEAR_FIELD_PHASES",
    "SOLVE_PHASES",
    "PhaseProfile",
    "active",
    "phase",
    "profiling",
]

#: FMM far-field stage phases, in the order ``drive_volume_fmm`` runs them.
FAR_FIELD_PHASES = (
    "far_form_multipoles",
    "far_coarsen_multipoles",
    "far_multipole_to_local",
    "far_eval_multipoles",
    "far_form_locals",
    "far_refine_locals",
    "far_eval_locals",
)

#: Near-field phases of one solve, in the order ``drive_volume_fmm`` runs
#: them.  ``split_correction`` is entered only by the online split
#: evaluator; a direct-table solve records zero seconds for it.
NEAR_FIELD_PHASES = (
    "nearfield_table_apply",
    "split_correction",
)

#: Every phase of one solve, in execution order.
SOLVE_PHASES = (
    "far_form_multipoles",
    "far_coarsen_multipoles",
    "nearfield_table_apply",
    "split_correction",
    "far_multipole_to_local",
    "far_eval_multipoles",
    "far_form_locals",
    "far_refine_locals",
    "far_eval_locals",
)


class PhaseProfile:
    """Accumulated wall seconds and entry counts, keyed by phase name.

    :arg sync: a callable invoked at every phase boundary to drain any
        asynchronous device queue (typically ``queue.finish``), or *None*
        for pure host-side code.
    """

    def __init__(self, *, sync=None):
        if sync is not None and not callable(sync):
            raise ValueError("sync must be callable or None")
        self._sync = sync
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._nested: set[str] = set()
        self._depth = 0

    # {{{ recording

    def sync(self) -> None:
        """Drain the device queue, if one was registered."""
        if self._sync is not None:
            self._sync()

    def record(self, name: str, seconds: float, *, calls: int = 1) -> None:
        """Add ``seconds`` (and ``calls`` entries) to phase ``name``."""
        seconds = float(seconds)
        if seconds < 0.0:
            raise ValueError("phase seconds must be >= 0")
        calls = int(calls)
        if calls < 0:
            raise ValueError("phase calls must be >= 0")
        name = str(name)
        self._seconds[name] = self._seconds.get(name, 0.0) + seconds
        self._calls[name] = self._calls.get(name, 0) + calls

    def _enter(self) -> None:
        self.sync()
        self._depth += 1

    def _exit(self, name: str, elapsed: float) -> None:
        self._depth -= 1
        if self._depth > 0:
            self._nested.add(str(name))
        self.record(name, elapsed)

    # }}}

    # {{{ readout

    @property
    def nested_names(self) -> frozenset[str]:
        """Phase names that were entered inside another phase.

        Non-empty means the recorded seconds double count and are not a
        partition of anything.
        """
        return frozenset(self._nested)

    def names(self) -> list[str]:
        """Recorded phase names, in first-recorded order."""
        return list(self._seconds)

    def seconds(self, name: str) -> float:
        """Total seconds recorded under ``name`` (0.0 if never entered)."""
        return float(self._seconds.get(str(name), 0.0))

    def calls(self, name: str) -> int:
        """Times ``name`` was entered (0 if never)."""
        return int(self._calls.get(str(name), 0))

    def mean_seconds(self, name: str) -> float:
        """Seconds per entry of ``name`` (0.0 if never entered)."""
        calls = self.calls(name)
        if calls == 0:
            return 0.0
        return self.seconds(name) / calls

    def total_seconds(self) -> float:
        """Sum over all recorded phases."""
        return float(sum(self._seconds.values()))

    def as_dict(self) -> dict[str, float]:
        """Copy of the accumulated seconds, keyed by phase name."""
        return dict(self._seconds)

    def shares(self, *, denominator: float | None = None) -> dict[str, float]:
        """Per-phase share of ``denominator`` (default: the phase total).

        Returns an empty mapping when the denominator is zero, rather than
        inventing a share.
        """
        total = (
            self.total_seconds() if denominator is None else float(denominator)
        )
        if total <= 0.0:
            return {}
        return {
            name: seconds / total for name, seconds in self._seconds.items()
        }

    # }}}


#: The profiles collecting in *this* execution context.
#:
#: A :class:`~contextvars.ContextVar` rather than a plain module global: the
#: activation has to follow the caller that opened it.  A second thread
#: running :func:`~volumential.volume_fmm.drive_volume_fmm` concurrently must
#: not have its work added to the first thread's profile, and -- worse, since
#: it silently corrupts the *other* solve's numbers too -- must not have its
#: phase boundaries drain the first thread's OpenCL queue instead of its own.
#: A fresh thread starts from the empty default; an :mod:`asyncio` task
#: inherits the context it was created in, which is the right answer for work
#: spawned inside a profiled block.  The value is an immutable tuple, so a
#: context that inherited it cannot mutate its parent's.
_ACTIVE: ContextVar[tuple[PhaseProfile, ...]] = ContextVar(
    "volumential_phase_profile_active", default=()
)


@contextmanager
def profiling(profile: PhaseProfile):
    """Activate ``profile`` for every :func:`phase` block in the body."""
    if not isinstance(profile, PhaseProfile):
        raise ValueError("profiling requires a PhaseProfile instance")
    token = _ACTIVE.set((*_ACTIVE.get(), profile))
    try:
        yield profile
    finally:
        # reset() restores the exact previous value, so nesting the same
        # profile twice unwinds correctly where list.remove() would not
        _ACTIVE.reset(token)


def active() -> bool:
    """Whether any profile is currently collecting in this context."""
    return bool(_ACTIVE.get())


class _InactivePhase:
    """The context manager :func:`phase` hands back when nothing is active.

    A single shared instance with empty ``__enter__``/``__exit__``, not a
    ``@contextmanager`` generator: entering a generator-based
    :class:`contextlib._GeneratorContextManager` costs about 1.4 us, which
    ``drive_volume_fmm`` would pay eight or nine times per *unprofiled*
    solve -- and eight versus nine, because only the online-split path
    enters ``split_correction``, so the "inert" instrumentation would add a
    strategy-dependent bias to exactly the solve means the break-even curves
    are built from.
    """

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_INACTIVE_PHASE = _InactivePhase()


class _ActivePhase:
    """Times one phase block against the profiles active on entry."""

    __slots__ = ("_name", "_profiles", "_start")

    def __init__(self, name, profiles):
        self._name = name
        self._profiles = profiles
        self._start = 0.0

    def __enter__(self):
        for profile in self._profiles:
            profile._enter()
        self._start = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        for profile in self._profiles:
            profile.sync()
        elapsed = time.perf_counter() - self._start
        for profile in self._profiles:
            profile._exit(str(self._name), elapsed)
        return False


def phase(name: str):
    """Time the body under phase ``name``; a no-op when nothing is active.

    Returns a context manager rather than being one, so the inactive path is
    a :class:`~contextvars.ContextVar` read and a shared-singleton return
    (see :class:`_InactivePhase` for why that matters).
    """
    profiles = _ACTIVE.get()
    if not profiles:
        return _INACTIVE_PHASE
    return _ActivePhase(name, profiles)
