"""Test move kernels: random walk and Galilean Monte Carlo.

For each move: (a) init returns valid state, (b) step produces expected output,
(c) JIT compilation succeeds, (d) vmap over walkers works, (e) acceptance
rate is reasonable on toy problem.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.moves.random_walk import build_kernel as rw_build_kernel
from jaxrens.sampling.moves.galilean import build_kernel as gmc_build_kernel
from jaxrens.state.mc_state import MCState, make_mc_state_class

# MCState with direction field for galilean tests
_GalileanMCState = make_mc_state_class({"direction": jnp.ndarray})


def _make_state(positions, types, energy, cell=None, step_size=0.1):
    """Helper: create MCState for non-galilean tests."""
    if cell is None:
        cell = jnp.zeros((3, 3))
    return MCState(
        positions=jnp.asarray(positions),
        types=jnp.asarray(types),
        energy=jnp.asarray(energy),
        cell=jnp.asarray(cell),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.array([step_size]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
    )


def _make_gmc_state(positions, types, energy, cell=None, step_size=0.1):
    """Helper: create MCState with direction for galilean tests."""
    if cell is None:
        cell = jnp.zeros((3, 3))
    return _GalileanMCState(
        positions=jnp.asarray(positions),
        types=jnp.asarray(types),
        energy=jnp.asarray(energy),
        cell=jnp.asarray(cell),
        step_size=jnp.asarray(step_size),
        step_sizes=jnp.array([step_size]),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
        direction=jnp.zeros_like(positions),
    )


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


@pytest.fixture
def positions():
    return jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])


@pytest.fixture
def types():
    return jnp.array([0, 0])


# ---------------------------------------------------------------------------
# Random Walk
# ---------------------------------------------------------------------------


class TestRandomWalkStep:
    def test_step_returns_state_and_info(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(positions, types, energy=1.0, step_size=0.05)
        step = jax.jit(rw_build_kernel(backend))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)

        assert isinstance(new_state, MCState)
        assert hasattr(info, "accepted")
        assert hasattr(info, "n_evaluations")

    def test_accepts_below_constraint(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(positions, types, energy=0.25, step_size=0.01)
        step = jax.jit(rw_build_kernel(backend))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=100.0)
        assert info.accepted

    def test_rejects_above_constraint(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(positions, types, energy=0.001, step_size=100.0)
        step = jax.jit(rw_build_kernel(backend))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.002)
        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)

    def test_jit(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(positions, types, energy=1.0, step_size=0.1)
        step = rw_build_kernel(backend)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 10.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, harmonic, positions, types):
        backend = harmonic
        step = jax.jit(rw_build_kernel(backend))

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([1.0, 2.0, 3.0, 4.0])
        batch_step_size = jnp.array([0.1, 0.1, 0.1, 0.1])

        batch_state = MCState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            cell=jnp.zeros((4, 3, 3)),
            step_size=batch_step_size,
            step_sizes=jnp.full((4, 1), 0.1),
            n_accepted=jnp.zeros((4, 1), dtype=jnp.int32),
            n_proposed=jnp.zeros((4, 1), dtype=jnp.int32),
            max_neighbor_count=jnp.zeros(4, dtype=jnp.int32),
            overflow=jnp.full(4, False),
            ensemble_params={},
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 10.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_acceptance_rate_reasonable(self, harmonic, positions, types):
        """Run 200 steps, expect reasonable acceptance rate on easy problem."""
        backend = harmonic
        state = _make_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(rw_build_kernel(backend))

        n_steps = 200
        n_accepted = 0
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=5.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        assert 0.1 < rate < 1.0, f"Acceptance rate {rate} out of expected range"

    def test_scan_compatible(self, harmonic, positions, types):
        """lax.scan should work with random walk step."""
        backend = harmonic
        state = _make_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(rw_build_kernel(backend))

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 5.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)


# ---------------------------------------------------------------------------
# Galilean Monte Carlo
# ---------------------------------------------------------------------------


class TestGalileanStep:
    def test_step_returns_state_and_info(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)

        assert hasattr(new_state, "positions")
        assert hasattr(new_state, "direction")
        assert hasattr(info, "accepted")

    def test_direction_initialized_on_first_call(self, harmonic, positions, types):
        """Direction should be randomized if initially zero."""
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        assert jnp.allclose(state.direction, 0.0)

        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))
        key = jax.random.key(0)
        new_state, _ = step(key, state, likelihood_constraint=10.0)

        dir_norm = jnp.sqrt(jnp.sum(new_state.direction**2))
        assert dir_norm > 0.1

    def test_jit(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = gmc_build_kernel(backend, n_reflect=3)

        jitted_step = jax.jit(step)
        key = jax.random.key(0)
        new_state, info = jitted_step(key, state, 10.0)
        assert new_state.energy.shape == ()

    def test_vmap(self, harmonic, positions, types):
        backend = harmonic
        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))

        batch_pos = jnp.stack([positions] * 4)
        batch_types = jnp.stack([types] * 4)
        batch_energy = jnp.array([0.25, 0.5, 0.75, 1.0])
        batch_step_size = jnp.full(4, 0.05)
        batch_dir = jnp.zeros((4, *positions.shape))

        batch_state = _GalileanMCState(
            positions=batch_pos,
            types=batch_types,
            energy=batch_energy,
            cell=jnp.zeros((4, 3, 3)),
            step_size=batch_step_size,
            step_sizes=jnp.full((4, 1), 0.05),
            n_accepted=jnp.zeros((4, 1), dtype=jnp.int32),
            n_proposed=jnp.zeros((4, 1), dtype=jnp.int32),
            max_neighbor_count=jnp.zeros(4, dtype=jnp.int32),
            overflow=jnp.full(4, False),
            ensemble_params={},
            direction=batch_dir,
        )

        keys = jax.random.split(jax.random.key(0), 4)
        vmapped_step = jax.vmap(step, in_axes=(0, 0, None))
        new_states, infos = vmapped_step(keys, batch_state, 10.0)

        assert new_states.energy.shape == (4,)
        assert infos.accepted.shape == (4,)

    def test_reflection_keeps_energy_below_constraint(self, harmonic, positions, types):
        """After GMC step, accepted states should have energy < Emax."""
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(backend, n_reflect=5))

        key = jax.random.key(42)
        for i in range(20):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=2.0)
            if info.accepted:
                assert state.energy < 2.0

    def test_acceptance_rate_reasonable(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(backend, n_reflect=5))

        n_steps = 100
        n_accepted = 0
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=2.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        assert 0.1 < rate <= 1.0, f"Acceptance rate {rate} out of expected range"

    def test_scan_compatible(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 2.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_without_forces(self, harmonic, positions, types):
        """use_forces=False should still work (random reflection)."""
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = gmc_build_kernel(
            backend, n_reflect=3, use_forces=False
        )

        key = jax.random.key(0)
        new_state, info = jax.jit(step)(key, state, 10.0)
        assert new_state.energy.shape == ()


class TestGalileanBaldockSemantics:
    """Pin the post-Phase-1 Baldock semantics: position advances through
    violations inside the scan; accept gates on FINAL energy only; NaN
    energies trigger reflection.  Mirrors jaxnest_dev/src/jaxnest/mcmc.py.

    The earlier in-tree variant reverted position+energy on every
    violation, making ``accepted`` trivially True (carry energy was
    always the last *good* energy, < Emax by construction) and silently
    propagating NaN energies as "valid".  These tests would have failed
    under that variant.
    """

    def test_reject_when_no_path_below_constraint(
        self, harmonic, positions, types,
    ):
        """With Emax STRICTLY below the initial state.energy, the
        trajectory cannot end below Emax — must report ``accepted=False``.

        Pre-fix this would have spuriously reported ``accepted=True``
        because the carry would have held the initial (below-Emax)
        energy throughout."""
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.5)
        # Emax BELOW state.energy — every reflection step lands at
        # E > Emax (harmonic potential at perturbed positions only grows).
        step = jax.jit(gmc_build_kernel(backend, n_reflect=4))
        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.1)
        assert not bool(info.accepted), (
            f"GMC must reject when state.energy ({float(state.energy)}) "
            f"already exceeds Emax (0.1); got accepted={bool(info.accepted)}"
        )
        # Reject path: position must revert to original.
        assert jnp.allclose(new_state.positions, state.positions)
        # Reject path: direction flips relative to state.direction (which
        # is zero on first call, so this just checks final_dir is not
        # used in the reject branch — position-invariance is the real
        # signal here).

    def test_accept_advances_through_intermediate_violations(
        self, harmonic, positions, types,
    ):
        """When ``n_reflect`` is large enough that reflections recover
        the walker, the move accepts AND the final position is NOT the
        initial position (i.e. the trajectory genuinely traversed the
        violated region rather than rejecting at the first step).
        Pre-fix this passed by luck (revert + early-Emax-check), but
        the new semantics make it a meaningful check."""
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(backend, n_reflect=5))
        key = jax.random.key(7)
        accept_count = 0
        moved_count = 0
        last_state = state
        for _ in range(50):
            key, sub = jax.random.split(key)
            last_state, info = step(sub, last_state, likelihood_constraint=0.5)
            if bool(info.accepted):
                accept_count += 1
                # Accepted → positions must differ from prior state by at
                # least ``step_size`` (since position advances unconditionally
                # inside the scan, then on accept we keep final_pos).
                # (We can't compare to the per-iter prior here without a
                # separate carry; use the moved counter as a proxy below.)
            moved_count += int(not jnp.allclose(last_state.positions, state.positions))
        assert accept_count > 5, (
            f"Expected nonzero acceptance under reasonable settings; got {accept_count}/50"
        )
        # Trajectory must have actually evolved (positions changed from
        # the original) — under the broken revert-on-violation kernel
        # walkers could "accept" with no displacement and this would fail.
        assert moved_count > 0

    def test_nan_energy_triggers_reflection_not_silent_reject(
        self, positions, types,
    ):
        """A backend that returns NaN at certain configurations must
        not cause the carry to propagate NaN energy through the scan
        and silently report ``accepted=False`` at the end (which is
        what happens without the NaN trap).  After the fix, NaN ⇒ inf
        ⇒ violation ⇒ reflection, exactly like an over-Emax energy.
        """
        # Synthetic backend: returns NaN at any displacement from
        # original positions, finite at original.
        positions = positions  # (2, 3)
        nan_check_pos = positions

        def nan_backend(p, types, cell, max_neighbors=0, ensemble_params=None):
            # Returns NaN whenever ANY coord differs from the initial
            # (forcing every proposed step to NaN-out).
            diff = jnp.any(jnp.abs(p - nan_check_pos) > 1e-6)
            e = jnp.where(diff, jnp.nan, 0.25)
            return BackendResult(energy=e)

        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(gmc_build_kernel(nan_backend, n_reflect=3, use_forces=False))
        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)
        # Every step NaN-violated → final energy = +inf → accepted=False
        # (NOT a NaN-corrupted "rejected" that silently passes through).
        assert not bool(info.accepted)
        # The reject path keeps state.positions and state.energy intact
        # — verify they survived a sea of NaN proposals.
        assert jnp.allclose(new_state.positions, state.positions)
        assert float(new_state.energy) == 0.25

    def test_final_energy_below_constraint_when_accepted(
        self, harmonic, positions, types,
    ):
        """When accepted, the stored state.energy must be < Emax.
        This is the post-fix invariant: accept ⇔ final-state energy
        below Emax.  Run for many trials to actually exercise the
        accept path."""
        backend = harmonic
        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.02)
        step = jax.jit(gmc_build_kernel(backend, n_reflect=5))
        key = jax.random.key(99)
        n_accepted = 0
        for _ in range(60):
            key, sub = jax.random.split(key)
            state, info = step(sub, state, likelihood_constraint=1.0)
            if bool(info.accepted):
                n_accepted += 1
                assert float(state.energy) < 1.0, (
                    f"Accepted move left state.energy={float(state.energy)} "
                    f"≥ Emax=1.0 — invariant violated"
                )
        assert n_accepted > 0  # sanity: the test actually exercised the accept path