"""Tests for BatchDescriptor and its three concrete implementations.

Coverage:
- SingleRun.split_keys: shape, bit-exact match with jax.random.split
- SingleRun.reduce_for_termination: identity on scalar inputs
- SingleRun.wrap_step: callable; runs on a fake ns_state
- VmapRuns.split_keys: shape (n_runs, n_sub_keys, 2); matches vmap(split)
- VmapRuns.reduce_for_termination: worst-of; scalar outputs
- VmapRuns.wrap_step: callable; runs vmapped on a fake batched ns_state
- PmapVmapRuns: all three methods raise NotImplementedError
- JIT compatibility for SingleRun and VmapRuns utility paths
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.sampling.batch_descriptor import (
    PmapVmapRuns,
    SingleRun,
    VmapRuns,
)


# ---------------------------------------------------------------------------
# Helpers — minimal fake ns_step and ns_state
# ---------------------------------------------------------------------------


def _fake_ns_step(ns_state, step_fn, n_mcmc_steps, n_extra):
    """Trivial ns_step replacement: identity on state, empty info dict."""
    return ns_state, {}


def _fake_ns_step_vmap(ns_state, step_fn, n_mcmc_steps, n_extra):
    """Same contract; used to verify vmap wrapping passes state through."""
    return ns_state, {}


class _FakeState:
    """Minimal JAX-pytree-compatible stand-in for NSState.

    Registers as a pytree so vmap can map over a batch of them.
    """
    def __init__(self, value: jax.Array):
        self.value = value

    def tree_flatten(self):
        return (self.value,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])


jax.tree_util.register_pytree_node_class(_FakeState)


# ---------------------------------------------------------------------------
# SingleRun tests
# ---------------------------------------------------------------------------


class TestSingleRun:
    def test_attributes(self):
        d = SingleRun()
        assert d.n_runs == 1
        assert d.shape_prefix == ()

    # --- split_keys ---

    def test_split_keys_shape(self):
        # With new-style typed keys, jax.random.split returns shape (n,) not (n, 2).
        d = SingleRun()
        key = jax.random.key(0)
        result = d.split_keys(key, 5)
        assert result.shape == (5,)

    def test_split_keys_bit_exact(self):
        d = SingleRun()
        key = jax.random.key(7)
        result = d.split_keys(key, 5)
        expected = jax.random.split(key, 5)
        # key arrays cannot be cast to np.ndarray directly; compare via key_data
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(result)),
            np.asarray(jax.random.key_data(expected)),
        )

    def test_split_keys_under_jit(self):
        d = SingleRun()
        key = jax.random.key(3)

        @jax.jit
        def _split(k):
            return d.split_keys(k, 4)

        result = _split(key)
        assert result.shape == (4,)
        expected = jax.random.split(key, 4)
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(result)),
            np.asarray(jax.random.key_data(expected)),
        )

    # --- reduce_for_termination ---

    def test_reduce_identity_on_scalars(self):
        d = SingleRun()
        log_ev = jnp.array(-5.3)
        hmax = jnp.array(2.1)
        ev_out, hmax_out = d.reduce_for_termination(log_ev, hmax)
        assert isinstance(ev_out, float)
        assert isinstance(hmax_out, float)
        assert abs(ev_out - (-5.3)) < 1e-6
        assert abs(hmax_out - 2.1) < 1e-6

    def test_reduce_identity_on_python_float(self):
        d = SingleRun()
        ev_out, hmax_out = d.reduce_for_termination(-3.0, 1.5)
        assert isinstance(ev_out, float)
        assert isinstance(hmax_out, float)
        assert abs(ev_out - (-3.0)) < 1e-9
        assert abs(hmax_out - 1.5) < 1e-9

    # --- wrap_step ---

    def test_wrap_step_returns_callable(self):
        d = SingleRun()
        jit_step = d.wrap_step(_fake_ns_step, None, 10, 0)
        assert callable(jit_step)

    def test_wrap_step_call_shape(self):
        """Calling the wrapped step returns (state, dict) unchanged."""
        d = SingleRun()
        jit_step = d.wrap_step(_fake_ns_step, None, 10, 0)
        state = _FakeState(jnp.array(42.0))
        out_state, out_info = jit_step(state, None, 10, 0)
        assert isinstance(out_state, _FakeState)
        assert isinstance(out_info, dict)
        np.testing.assert_allclose(float(out_state.value), 42.0)

    def test_wrap_step_under_jit(self):
        """Wrapping via SingleRun must be JIT-safe — no tracing issues."""
        d = SingleRun()
        jit_step = d.wrap_step(_fake_ns_step, None, 5, 0)
        # The returned object is already jit'd; calling it exercises tracing.
        state = _FakeState(jnp.array(0.0))
        out_state, _ = jit_step(state, None, 5, 0)
        assert out_state is not None


# ---------------------------------------------------------------------------
# VmapRuns tests
# ---------------------------------------------------------------------------


class TestVmapRuns:
    def test_attributes(self):
        d = VmapRuns(n_runs=3)
        assert d.n_runs == 3
        assert d.shape_prefix == (3,)

    # --- split_keys ---

    def test_split_keys_shape(self):
        # With new-style typed keys, jax.random.split returns shape (n,) not (n, 2).
        d = VmapRuns(n_runs=3)
        base_key = jax.random.key(0)
        run_keys = jax.random.split(base_key, 3)  # (3,)
        result = d.split_keys(run_keys, 5)
        assert result.shape == (3, 5)

    def test_split_keys_matches_vmap(self):
        """Must match vmap(jax.random.split) applied per-run."""
        d = VmapRuns(n_runs=3)
        base_key = jax.random.key(42)
        run_keys = jax.random.split(base_key, 3)
        result = d.split_keys(run_keys, 5)
        expected = jax.vmap(jax.random.split, in_axes=(0, None))(run_keys, 5)
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(result)),
            np.asarray(jax.random.key_data(expected)),
        )

    def test_split_keys_under_jit(self):
        d = VmapRuns(n_runs=4)
        run_keys = jax.random.split(jax.random.key(1), 4)

        @jax.jit
        def _split(keys):
            return d.split_keys(keys, 3)

        result = _split(run_keys)
        assert result.shape == (4, 3)

    # --- reduce_for_termination ---

    def test_reduce_worst_of(self):
        d = VmapRuns(n_runs=3)
        log_ev = jnp.array([-10.0, -5.0, -7.0])
        hmax = jnp.array([2.0, 5.0, 3.0])
        ev_out, hmax_out = d.reduce_for_termination(log_ev, hmax)
        assert isinstance(ev_out, float)
        assert isinstance(hmax_out, float)
        assert abs(ev_out - (-10.0)) < 1e-6   # min(log_evidence)
        assert abs(hmax_out - 5.0) < 1e-6     # max(hmax)

    def test_reduce_scalar_outputs(self):
        d = VmapRuns(n_runs=5)
        log_ev = jnp.zeros(5)
        hmax = jnp.ones(5)
        ev_out, hmax_out = d.reduce_for_termination(log_ev, hmax)
        assert isinstance(ev_out, float)
        assert isinstance(hmax_out, float)

    def test_reduce_single_run_edge(self):
        """VmapRuns(n_runs=1) reduce is equivalent to SingleRun reduce."""
        d_vmap = VmapRuns(n_runs=1)
        d_single = SingleRun()
        log_ev_v = jnp.array([-3.5])
        hmax_v = jnp.array([1.2])
        log_ev_s = jnp.array(-3.5)
        hmax_s = jnp.array(1.2)
        ev_v, h_v = d_vmap.reduce_for_termination(log_ev_v, hmax_v)
        ev_s, h_s = d_single.reduce_for_termination(log_ev_s, hmax_s)
        assert abs(ev_v - ev_s) < 1e-6
        assert abs(h_v - h_s) < 1e-6

    # --- wrap_step ---

    def test_wrap_step_returns_callable(self):
        d = VmapRuns(n_runs=2)
        jit_step = d.wrap_step(_fake_ns_step_vmap, None, 5, 0)
        assert callable(jit_step)

    def test_wrap_step_call_batched(self):
        """Wrapped step must vmap over a batch of states."""
        d = VmapRuns(n_runs=4)
        jit_step = d.wrap_step(_fake_ns_step_vmap, None, 5, 0)
        # Batch of 4 fake states
        batched_state = _FakeState(jnp.arange(4, dtype=jnp.float32))
        out_states, out_infos = jit_step(batched_state)
        assert isinstance(out_states, _FakeState)
        # vmap over value dim should give shape (4,)
        assert out_states.value.shape == (4,)

    def test_wrap_step_under_jit(self):
        d = VmapRuns(n_runs=3)
        jit_step = d.wrap_step(_fake_ns_step_vmap, None, 5, 0)
        batched_state = _FakeState(jnp.ones(3))
        out_states, _ = jit_step(batched_state)
        assert out_states.value.shape == (3,)


# ---------------------------------------------------------------------------
# PmapVmapRuns tests
# ---------------------------------------------------------------------------


class TestPmapVmapRuns:
    def test_attributes(self):
        d = PmapVmapRuns(n_gpu=2, n_per_gpu=4)
        assert d.n_runs == 8
        assert d.shape_prefix == (2, 4)

    def test_is_batched(self):
        d = PmapVmapRuns(n_gpu=1, n_per_gpu=3)
        assert d.is_batched is True

    def test_n_runs_computed_correctly(self):
        for n_gpu, n_per in [(1, 1), (2, 4), (4, 8)]:
            d = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per)
            assert d.n_runs == n_gpu * n_per
            assert d.shape_prefix == (n_gpu, n_per)

    # --- split_keys ---

    def test_split_keys_shape(self):
        """split_keys returns shape (G, P, n_sub_keys) with typed-key dtype."""
        n_gpu, n_per_gpu = 1, 3
        d = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
        base_key = jax.random.key(0)
        gpu_keys = jax.random.split(base_key, n_gpu)  # (G,)
        result = d.split_keys(gpu_keys, 5)
        assert result.shape == (n_gpu, n_per_gpu, 5)

    def test_split_keys_deterministic(self):
        """Two calls with the same key produce identical results."""
        n_gpu, n_per_gpu = 1, 2
        d = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
        gpu_keys = jax.random.split(jax.random.key(7), n_gpu)
        r1 = d.split_keys(gpu_keys, 4)
        r2 = d.split_keys(gpu_keys, 4)
        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(r1)),
            np.asarray(jax.random.key_data(r2)),
        )

    def test_split_keys_under_jit(self):
        n_gpu, n_per_gpu = 1, 3
        d = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
        gpu_keys = jax.random.split(jax.random.key(3), n_gpu)

        @jax.jit
        def _split(keys):
            return d.split_keys(keys, 4)

        result = _split(gpu_keys)
        assert result.shape == (n_gpu, n_per_gpu, 4)

    # --- reduce_for_termination ---

    def test_reduce_worst_of_2d(self):
        """reduce_for_termination on (G, P) input returns worst-of scalars."""
        d = PmapVmapRuns(n_gpu=1, n_per_gpu=3)
        log_ev = jnp.array([[-10.0, -5.0, -7.0]])  # (1, 3)
        hmax = jnp.array([[2.0, 5.0, 3.0]])         # (1, 3)
        ev_out, hmax_out = d.reduce_for_termination(log_ev, hmax)
        assert isinstance(ev_out, float)
        assert isinstance(hmax_out, float)
        assert abs(ev_out - (-10.0)) < 1e-6   # min across all
        assert abs(hmax_out - 5.0) < 1e-6     # max across all

    def test_reduce_scalar_outputs(self):
        d = PmapVmapRuns(n_gpu=1, n_per_gpu=4)
        log_ev = jnp.zeros((1, 4))
        hmax = jnp.ones((1, 4))
        ev_out, hmax_out = d.reduce_for_termination(log_ev, hmax)
        assert isinstance(ev_out, float)
        assert isinstance(hmax_out, float)

    # --- wrap_step ---

    def test_wrap_step_returns_callable(self):
        d = PmapVmapRuns(n_gpu=1, n_per_gpu=2)
        jit_step = d.wrap_step(_fake_ns_step_vmap, None, 5, 0)
        assert callable(jit_step)

    def test_wrap_step_call_pmap_vmap(self):
        """wrap_step on (G, P)-batched fake state runs without error."""
        n_gpu, n_per_gpu = 1, 2
        d = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)
        jit_step = d.wrap_step(_fake_ns_step_vmap, None, 5, 0)
        # (G, P) batch — pmap maps over G, vmap maps over P.
        batched_state = _FakeState(jnp.ones((n_gpu, n_per_gpu)))
        out_states, out_infos = jit_step(batched_state)
        assert isinstance(out_states, _FakeState)
        assert out_states.value.shape == (n_gpu, n_per_gpu)
