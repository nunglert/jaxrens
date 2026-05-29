"""Tests for hysteresis-gated bucket shrinking in the NS outer loop.

Covers:
- ``_pick_prev_bucket`` — pure-Python picker that returns the next-smaller
  ladder entry when it still safely accommodates the observed peak plus
  offset.
- ``_run_loop`` integration — shrinks fire only after the dwell window,
  a single high spike resets the streak, ``shrink_dwell=0`` keeps the
  bucket pinned, and an overflow growth invalidates a pending shrink.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxrens.sampling.moves.random_walk as _rw_mod
from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import SingleRun
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns
from jaxrens.sampling.bucket_manager import BucketManager, _pick_prev_bucket
from jaxrens.sampling.run_loop import _run_loop
from jaxrens.sampling.termination import IterationTermination


def _rw_descriptor(step_size: float = 0.3) -> MoveKernel:
    return MoveKernel(
        name="random_walk",
        build_kernel=_rw_mod.build_kernel,
        step_size=step_size,
        weight=1.0,
        kernel_kwargs={},
        extra_state_fields={},
    )


def _build_ns_state(n_walkers: int = 6, n_atoms: int = 2, seed: int = 0):
    """Minimal NSState backed by the harmonic toy backend + random-walk move."""
    backend = create_harmonic()
    desc = _rw_descriptor()
    init_fn, step_fn, _per_move_fns = build_mwg(backend, [desc])

    key = jax.random.key(seed)
    key, key_pos = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n_walkers, n_atoms, 3), minval=-2.0, maxval=2.0,
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    ns_state = init_ns(init_fn, positions, types, energies, cells, key)
    return ns_state, step_fn, backend


# ---------------------------------------------------------------------------
# Pure-Python picker tests
# ---------------------------------------------------------------------------


class TestPickPrevBucket:
    """``_pick_prev_bucket`` returns the largest ladder entry strictly below
    ``current`` that still satisfies ``entry >= true_max + offset``,
    or ``None`` when no smaller entry qualifies."""

    LADDER = (30, 35, 40, 45, 50)

    def test_returns_next_smaller_when_headroom_available(self):
        # current=50, observed peak well below 45 → step down to 45.
        assert _pick_prev_bucket(true_max=20, current=50,
                                 ladder=self.LADDER,
                                 offset=5) == 45

    def test_returns_none_when_target_requires_current_or_larger(self):
        # observed + offset = 35 + 5 = 40.  Largest entry strictly below
        # current=40 is 35 < 40 → no qualifying entry → None.
        assert _pick_prev_bucket(true_max=35, current=40,
                                 ladder=self.LADDER,
                                 offset=5) is None

    def test_returns_none_when_current_is_smallest_entry(self):
        # current=30 is the bottom of the ladder — nothing smaller exists.
        assert _pick_prev_bucket(true_max=5, current=30,
                                 ladder=self.LADDER,
                                 offset=5) is None

    def test_steps_down_one_ladder_entry_even_if_much_smaller_qualifies(self):
        # observed peak well below 30 — picker still only returns the
        # immediate-next-smaller entry (45), never a multi-step jump.
        assert _pick_prev_bucket(true_max=5, current=50,
                                 ladder=self.LADDER,
                                 offset=0) == 45

    def test_offset_acts_as_shrink_safety_margin(self):
        # With offset=5, true_max=25 → target=30 → 35 qualifies (next-smaller).
        assert _pick_prev_bucket(true_max=25, current=40,
                                 ladder=self.LADDER,
                                 offset=5) == 35
        # With offset=11, true_max=25 → target=36 → 35 < 36 → None.
        assert _pick_prev_bucket(true_max=25, current=40,
                                 ladder=self.LADDER,
                                 offset=11) is None

    def test_zero_offset_picks_immediate_neighbour(self):
        # current=45, target=true_max=40 → 40 qualifies → return 40.
        assert _pick_prev_bucket(true_max=40, current=45,
                                 ladder=self.LADDER,
                                 offset=0) == 40


# ---------------------------------------------------------------------------
# Loop-level integration tests
# ---------------------------------------------------------------------------


def _wrap_step_fn_force_observed(real_step_fn, observed_max: int):
    """Wrap ``real_step_fn`` so its output reports a fixed observed peak.

    The wrapper overrides ``max_neighbor_count`` after each step so that the
    outer loop's shrink-check sees the value we want, independent of whatever
    the real harmonic backend would otherwise report.  ``overflow`` is forced
    to ``False`` so the upgrade path never fires.
    """
    def wrapped(key, state, emax):
        new_state, info = real_step_fn(key, state, emax)
        new_state = new_state.set(
            max_neighbor_count=jnp.asarray(
                observed_max, dtype=new_state.max_neighbor_count.dtype,
            ),
            overflow=jnp.asarray(False),
        )
        return new_state, info
    return wrapped


def _run_loop_minimal(ns_state, step_fn, *, n_iters: int,
                      shrink_dwell: int = 0,
                      ladder=(30, 35, 40, 45, 50),
                      offset: int = 0):
    """Drive ``_run_loop`` with a SingleRun batcher and an iteration cap.

    Mirrors the minimal slice of ``run_ns`` needed to exercise the
    shrink-check block — no adaptation, no callbacks, no inter-RE.
    """
    batcher = SingleRun()
    adapt_step = build_adapt_step(
        move_descriptors=[],
        per_move_fns=None,
        batcher=batcher,
        adjust_n_samples=1,
        adjust_factor=1.5,
        adjust_max_rounds=1,
        adjust_interval=0,
    )
    ns_state_out, _rng, _cum = _run_loop(
        batcher=batcher,
        adapt_step=adapt_step,
        adjust_interval=0,
        ns_state=ns_state,
        step_fn=step_fn,
        n_mcmc_steps=1,
        n_extra=0,
        termination_criteria=[IterationTermination(n_iters)],
        callbacks=[],
        n_moves=1,
        move_descriptors=[_rw_descriptor()],
        rng_key=jax.random.key(123),
        info_interval=1,
        max_neighbors_list=ladder,
        max_neighbors_offset=offset,
        max_neighbors_shrink_dwell=shrink_dwell,
    )
    return ns_state_out


class TestBucketShrinkInLoop:
    """End-to-end behaviour of the shrink-check block inside ``_run_loop``."""

    LADDER = (30, 35, 40, 45, 50)

    def test_disabled_when_dwell_is_zero(self):
        """``shrink_dwell=0`` keeps the bucket pinned, even with low observed peaks."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=5)

        result = _run_loop_minimal(
            ns_state, wrapped, n_iters=10,
            shrink_dwell=0, ladder=self.LADDER, offset=0,
        )
        assert int(result.population.max_neighbors) == 50

    def test_shrinks_once_after_dwell_window(self):
        """After ``shrink_dwell`` low iterations, the bucket steps down one entry."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=5)

        # dwell=3, offset=0 → after 3 low iterations the picker returns
        # the next-smaller entry (45) and the bucket is shrunk once.
        result = _run_loop_minimal(
            ns_state, wrapped, n_iters=4,
            shrink_dwell=3, ladder=self.LADDER, offset=0,
        )
        assert int(result.population.max_neighbors) == 45

    def test_shrinks_multiple_steps_over_consecutive_windows(self):
        """Each shrink resets the counter; sustained low peaks step down repeatedly."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=5)

        # dwell=2 → 50→45 at iter 1, 45→40 at iter 3, 40→35 at iter 5,
        # 35→30 at iter 7.  After 9 iterations the bucket is at 30.
        result = _run_loop_minimal(
            ns_state, wrapped, n_iters=9,
            shrink_dwell=2, ladder=self.LADDER, offset=0,
        )
        assert int(result.population.max_neighbors) == 30

    def test_offset_can_block_shrink_at_boundary(self):
        """The offset slack prevents shrinking when the observed peak sits
        right at the next-smaller entry."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        # observed=45 with offset=0 → target=45 fits in 45 → shrink fires.
        # With offset=1 → target=46 > 45 → no qualifying entry → no shrink.
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=45)
        result = _run_loop_minimal(
            ns_state, wrapped, n_iters=10,
            shrink_dwell=2, ladder=self.LADDER, offset=1,
        )
        assert int(result.population.max_neighbors) == 50


class TestBucketManagerDirect:
    """Direct unit tests on the BucketManager state machine, decoupled from
    the outer NS / burn-in loops.  Verifies the counter resets and shrink
    cadence using a stub state object — no JAX needed beyond jnp scalars."""

    LADDER = (30, 35, 40, 45, 50)

    @staticmethod
    def _stub_state(bucket: int, observed: int, overflow: bool = False):
        """Build a minimal duck-typed ns_state with the fields BucketManager reads."""

        class _Pop:
            def __init__(self, b, o, of):
                self.max_neighbors = b
                self.max_neighbor_count = jnp.asarray([o], dtype=jnp.int32)
                self.overflow = jnp.asarray(of)

            def set(self, **kw):
                new = _Pop(self.max_neighbors, int(self.max_neighbor_count[0]),
                           bool(self.overflow))
                for k, v in kw.items():
                    setattr(new, k, v)
                return new

        class _State:
            def __init__(self, pop):
                self.population = pop

            def set(self, **kw):
                new = _State(self.population)
                for k, v in kw.items():
                    setattr(new, k, v)
                return new

        return _State(_Pop(bucket, observed, overflow))

    def test_growth_resets_low_count(self):
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=3)
        # Build up a low streak (starting bucket=40 so an upward grow has
        # somewhere to go).  observed=5 + offset=0 leaves plenty of headroom
        # for the next-smaller pick.
        st = self._stub_state(bucket=40, observed=5)
        st = mgr.maybe_shrink(st, iteration=0)
        st = mgr.maybe_shrink(st, iteration=1)
        assert mgr.low_count == 2
        # Now signal overflow on the next step — observed=42 needs bucket>=42
        # AND >current=40 → picker returns 45.
        new = self._stub_state(bucket=40, observed=42, overflow=True)
        retried, retry = mgr.grow_if_overflow(
            st, new, label="iter", iteration=2,
        )
        assert retry is True
        assert int(retried.population.max_neighbors) == 45
        # Most important invariant: growing wiped the streak.
        assert mgr.low_count == 0

    def test_spike_resets_streak(self):
        """A single high observed peak resets ``low_count`` mid-streak.

        Drives ``BucketManager.maybe_shrink`` directly with three observed
        peaks (5, 46, 5) under ``dwell=3, offset=0`` on the (30,35,40,45,50)
        ladder starting at bucket=50.  Iter 0 (obs=5) increments the streak
        to 1; iter 1 (obs=46) fails the check against the next-smaller bucket
        45 and resets the streak to 0; iter 2 (obs=5) raises it to 1 again.
        Net: no shrink, bucket stays at 50.

        Implemented as a direct ``BucketManager`` drive rather than as a
        ``_run_loop`` integration to sidestep ``jax.jit`` tracing — Python
        side-channel state in a wrapped ``step_fn`` does not survive
        compilation.
        """
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=3)
        st = self._stub_state(bucket=50, observed=5)
        st = mgr.maybe_shrink(st, iteration=0)
        assert mgr.low_count == 1
        st = self._stub_state(bucket=int(st.population.max_neighbors), observed=46)
        st = mgr.maybe_shrink(st, iteration=1)
        assert mgr.low_count == 0
        st = self._stub_state(bucket=int(st.population.max_neighbors), observed=5)
        st = mgr.maybe_shrink(st, iteration=2)
        assert mgr.low_count == 1
        assert int(st.population.max_neighbors) == 50

    def test_shrink_fires_after_dwell(self):
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=2)
        st = self._stub_state(bucket=50, observed=5)
        st = mgr.maybe_shrink(st, iteration=0)
        assert int(st.population.max_neighbors) == 50
        st = mgr.maybe_shrink(st, iteration=1)
        # dwell=2 reached; bucket steps one entry down to 45.
        assert int(st.population.max_neighbors) == 45
        assert mgr.low_count == 0

    def test_no_overflow_is_passthrough(self):
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=0)
        old = self._stub_state(bucket=40, observed=5, overflow=False)
        new = self._stub_state(bucket=40, observed=12, overflow=False)
        out, retry = mgr.grow_if_overflow(old, new, label="iter", iteration=0)
        assert retry is False
        assert out is new


class TestNegativeParamsRejected:
    """Out-of-range shrink params are caught before the loop starts."""

    def test_negative_dwell_raises(self):
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        with pytest.raises(ValueError, match="shrink_dwell"):
            _run_loop_minimal(
                ns_state, step_fn, n_iters=1,
                shrink_dwell=-1,
            )
