"""Tests for hysteresis-gated bucket shrinking in the NS outer loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxrens.sampling.moves.random_walk as _rw_mod
from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.adaptation.manager import build_adapt_step
from jaxrens.sampling.batch_descriptor import SingleRun
from jaxrens.sampling.bucket_manager import BucketManager, _pick_prev_bucket
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns
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
    """Return a small NSState on the harmonic backend with a random walk."""
    backend = create_harmonic()
    desc = _rw_descriptor()
    init_fn, step_fn, _per_move_fns = build_mwg(backend, [desc])

    key = jax.random.key(seed)
    key, key_pos = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n_walkers, n_atoms, 3),
        minval=-2.0,
        maxval=2.0,
    )
    types = jnp.zeros((n_atoms,), dtype=jnp.int32)
    cells = jnp.zeros((n_walkers, 3, 3))
    energies = jax.vmap(
        lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

    ns_state = init_ns(init_fn, positions, types, energies, cells, key)
    return ns_state, step_fn, backend


class TestPickPrevBucket:
    """Check how _pick_prev_bucket chooses the next-smaller ladder entry.

    It returns the largest entry strictly below ``current`` that still
    satisfies ``entry >= true_max + offset``, or ``None`` when none does.
    """

    LADDER = (30, 35, 40, 45, 50)

    def test_returns_next_smaller_when_headroom_available(self):
        assert (
            _pick_prev_bucket(
                true_max=20, current=50, ladder=self.LADDER, offset=5
            )
            == 45
        )

    def test_returns_none_when_target_requires_current_or_larger(self):
        # target = 35 + 5 = 40, but the largest entry below current=40 is 35.
        assert (
            _pick_prev_bucket(
                true_max=35, current=40, ladder=self.LADDER, offset=5
            )
            is None
        )

    def test_returns_none_when_current_is_smallest_entry(self):
        assert (
            _pick_prev_bucket(
                true_max=5, current=30, ladder=self.LADDER, offset=5
            )
            is None
        )

    def test_steps_down_one_ladder_entry_even_if_much_smaller_qualifies(self):
        # Even a peak far below 30 gives only the next entry, never a jump.
        assert (
            _pick_prev_bucket(
                true_max=5, current=50, ladder=self.LADDER, offset=0
            )
            == 45
        )

    def test_offset_acts_as_shrink_safety_margin(self):
        # offset=5: target=30, so 35 qualifies.
        assert (
            _pick_prev_bucket(
                true_max=25, current=40, ladder=self.LADDER, offset=5
            )
            == 35
        )
        # offset=11: target=36 > 35, so nothing qualifies.
        assert (
            _pick_prev_bucket(
                true_max=25, current=40, ladder=self.LADDER, offset=11
            )
            is None
        )

    def test_zero_offset_picks_immediate_neighbour(self):
        assert (
            _pick_prev_bucket(
                true_max=40, current=45, ladder=self.LADDER, offset=0
            )
            == 40
        )


def _wrap_step_fn_force_observed(real_step_fn, observed_max: int):
    """Wrap ``real_step_fn`` so each step reports a fixed observed peak.

    ``max_neighbor_count`` is overridden after every step and ``overflow`` is
    forced to ``False``, so only the shrink path can fire.
    """

    def wrapped(key, state, emax, move_key):
        new_state, info = real_step_fn(key, state, emax, move_key)
        new_state = new_state.set(
            max_neighbor_count=jnp.asarray(
                observed_max,
                dtype=new_state.max_neighbor_count.dtype,
            ),
            overflow=jnp.asarray(False),
        )
        return new_state, info

    return wrapped


def _run_loop_minimal(
    ns_state,
    step_fn,
    *,
    n_iters: int,
    shrink_dwell: int = 0,
    ladder=(30, 35, 40, 45, 50),
    offset: int = 0,
):
    """Run ``_run_loop`` for ``n_iters`` iterations with no adaptation."""
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
    """Check the shrink block of _run_loop on a harmonic population.

    Shrinks fire only after the dwell window, ``shrink_dwell=0`` keeps the
    bucket pinned, and the offset can block a shrink at the boundary.
    """

    LADDER = (30, 35, 40, 45, 50)

    def test_disabled_when_dwell_is_zero(self):
        """``shrink_dwell=0`` pins the bucket even with low observed peaks."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=5)

        result = _run_loop_minimal(
            ns_state,
            wrapped,
            n_iters=10,
            shrink_dwell=0,
            ladder=self.LADDER,
            offset=0,
        )
        assert int(result.population.max_neighbors) == 50

    def test_shrinks_once_after_dwell_window(self):
        """After ``shrink_dwell`` low iterations the bucket steps down once."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=5)

        result = _run_loop_minimal(
            ns_state,
            wrapped,
            n_iters=4,
            shrink_dwell=3,
            ladder=self.LADDER,
            offset=0,
        )
        assert int(result.population.max_neighbors) == 45

    def test_shrinks_multiple_steps_over_consecutive_windows(self):
        """Each shrink resets the counter, so low peaks keep stepping down."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=5)

        # dwell=2: 50->45 at iter 1, 45->40 at iter 3, 40->35 at iter 5,
        # 35->30 at iter 7.
        result = _run_loop_minimal(
            ns_state,
            wrapped,
            n_iters=9,
            shrink_dwell=2,
            ladder=self.LADDER,
            offset=0,
        )
        assert int(result.population.max_neighbors) == 30

    def test_offset_can_block_shrink_at_boundary(self):
        """The offset stops a shrink when the peak sits at the next entry."""
        ns_state, step_fn, _backend = _build_ns_state(n_walkers=4, n_atoms=2)
        ns_state = ns_state.set(
            population=ns_state.population.set(max_neighbors=50),
        )
        # observed=45: offset=0 would shrink to 45, offset=1 needs 46.
        wrapped = _wrap_step_fn_force_observed(step_fn, observed_max=45)
        result = _run_loop_minimal(
            ns_state,
            wrapped,
            n_iters=10,
            shrink_dwell=2,
            ladder=self.LADDER,
            offset=1,
        )
        assert int(result.population.max_neighbors) == 50


class TestBucketManagerDirect:
    """Check the BucketManager counter and shrink cadence on stub states."""

    LADDER = (30, 35, 40, 45, 50)

    @staticmethod
    def _stub_state(bucket: int, observed: int, overflow: bool = False):
        """Build a duck-typed ns_state with the fields BucketManager reads."""

        class _Pop:
            def __init__(self, b, o, of):
                self.max_neighbors = b
                self.max_neighbor_count = jnp.asarray([o], dtype=jnp.int32)
                self.overflow = jnp.asarray(of)

            def set(self, **kw):
                new = _Pop(
                    self.max_neighbors,
                    int(self.max_neighbor_count[0]),
                    bool(self.overflow),
                )
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
        st = self._stub_state(bucket=40, observed=5)
        st = mgr.maybe_shrink(st, iteration=0)
        st = mgr.maybe_shrink(st, iteration=1)
        assert mgr.low_count == 2
        # observed=42 overflows bucket 40, so the manager grows to 45.
        new = self._stub_state(bucket=40, observed=42, overflow=True)
        retried, retry = mgr.grow_if_overflow(
            st,
            new,
            label="iter",
            iteration=2,
        )
        assert retry is True
        assert int(retried.population.max_neighbors) == 45
        assert mgr.low_count == 0

    def test_spike_resets_streak(self):
        """A single high peak (5, 46, 5) resets the low streak; no shrink.

        The manager is driven directly because Python side effects in a
        wrapped ``step_fn`` do not survive ``jax.jit``.
        """
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=3)
        st = self._stub_state(bucket=50, observed=5)
        st = mgr.maybe_shrink(st, iteration=0)
        assert mgr.low_count == 1
        st = self._stub_state(
            bucket=int(st.population.max_neighbors), observed=46
        )
        st = mgr.maybe_shrink(st, iteration=1)
        assert mgr.low_count == 0
        st = self._stub_state(
            bucket=int(st.population.max_neighbors), observed=5
        )
        st = mgr.maybe_shrink(st, iteration=2)
        assert mgr.low_count == 1
        assert int(st.population.max_neighbors) == 50

    def test_shrink_fires_after_dwell(self):
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=2)
        st = self._stub_state(bucket=50, observed=5)
        st = mgr.maybe_shrink(st, iteration=0)
        assert int(st.population.max_neighbors) == 50
        st = mgr.maybe_shrink(st, iteration=1)
        assert int(st.population.max_neighbors) == 45
        assert mgr.low_count == 0

    def test_no_overflow_is_passthrough(self):
        mgr = BucketManager(self.LADDER, offset=0, shrink_dwell=0)
        old = self._stub_state(bucket=40, observed=5, overflow=False)
        new = self._stub_state(bucket=40, observed=12, overflow=False)
        out, retry = mgr.grow_if_overflow(old, new, label="iter", iteration=0)
        assert retry is False
        assert out is new

    def test_negative_dwell_raises(self):
        with pytest.raises(ValueError, match="shrink_dwell"):
            BucketManager(self.LADDER, offset=0, shrink_dwell=-1)

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError, match="ladder is empty"):
            BucketManager((), offset=0)
