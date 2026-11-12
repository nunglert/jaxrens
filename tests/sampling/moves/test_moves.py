"""Tests for the random walk and Galilean Monte Carlo move kernels."""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.moves import GMCMoveSpec
from jaxrens.sampling.moves.galilean import _perturb_direction
from jaxrens.sampling.moves.galilean import build_kernel as gmc_build_kernel
from jaxrens.sampling.moves.random_walk import build_kernel as rw_build_kernel
from jaxrens.sampling.mwg import build_mwg
from jaxrens.state.mc_state import MCState, make_mc_state_class

_GalileanMCState = make_mc_state_class({"direction": jnp.ndarray})

_SYMBOL_MAP = {0: "Ge", 1: "Si"}


def _make_state(positions, types, energy, cell=None, step_size=0.1):
    """Build an MCState for one walker."""
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


def _make_gmc_state(
    positions,
    types,
    energy,
    cell=None,
    step_size=0.1,
    direction_field="direction",
):
    """Build a Galilean MCState for one walker, with a zero direction."""
    if cell is None:
        cell = jnp.zeros((3, 3))
    if direction_field == "direction":
        state_cls = _GalileanMCState
    else:
        state_cls = make_mc_state_class({direction_field: jnp.ndarray})
    return state_cls(
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
        **{direction_field: jnp.zeros_like(positions)},
    )


def _energy(positions):
    """Harmonic energy (k=1) of the positions."""
    return 0.5 * jnp.sum(positions**2)


@pytest.fixture
def harmonic():
    return create_harmonic(k=1.0)


@pytest.fixture
def positions():
    return jnp.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.3, 0.3, 0.0],
        ]
    )


@pytest.fixture
def types():
    """Atoms 0, 1 have type 0 and atoms 2, 3 have type 1."""
    return jnp.array([0, 0, 1, 1])


class TestRandomWalkStep:
    """Check the random walk kernel on a harmonic potential.

    The tests cover one step, accept and reject against the constraint, vmap,
    lax.scan, and the acceptance rate.
    """

    def test_step_returns_state_and_info(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(positions, types, energy=1.0, step_size=0.05)
        step = jax.jit(rw_build_kernel(backend))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)

        assert isinstance(new_state, MCState)
        assert new_state.positions.shape == positions.shape
        assert new_state.energy.shape == ()
        assert info.n_evaluations == 1

    def test_accepts_below_constraint(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(
            positions, types, energy=_energy(positions), step_size=0.01
        )
        step = jax.jit(rw_build_kernel(backend))

        key = jax.random.key(42)
        new_state, info = step(key, state, likelihood_constraint=100.0)

        assert info.accepted
        expected = backend(new_state.positions, types, state.cell, 0)[0]
        assert jnp.allclose(new_state.energy, expected)

    def test_rejects_above_constraint(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(positions, types, energy=0.001, step_size=100.0)
        step = jax.jit(rw_build_kernel(backend))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=0.002)
        assert not info.accepted
        assert jnp.array_equal(new_state.positions, state.positions)

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
        backend = harmonic
        state = _make_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = jax.jit(rw_build_kernel(backend))

        n_steps = 200
        n_accepted = 0
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=5.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        assert 0.1 < rate < 1.0

    def test_scan_compatible(self, harmonic, positions, types):
        backend = harmonic
        state = _make_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = jax.jit(rw_build_kernel(backend))

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 5.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)


class TestGalileanStep:
    """Check the Galilean kernel on a harmonic potential.

    The tests cover one step, the direction (initial draw and refresh), vmap,
    lax.scan, the acceptance rate, and the energy constraint.
    """

    def test_step_returns_state_and_info(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))

        key = jax.random.key(0)
        new_state, info = step(key, state, likelihood_constraint=10.0)

        assert new_state.direction.shape == positions.shape
        assert new_state.energy.shape == ()
        assert info.n_evaluations == 3
        assert info.n_grad_evaluations == 3

    def test_direction_initialized_on_first_call(
        self, harmonic, positions, types
    ):
        """A zero stored direction is replaced by a random one."""
        backend = harmonic
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        assert jnp.allclose(state.direction, 0.0)

        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))
        key = jax.random.key(0)
        new_state, _ = step(key, state, likelihood_constraint=10.0)

        dir_norm = jnp.sqrt(jnp.sum(new_state.direction**2))
        assert dir_norm > 0.1

    @pytest.mark.parametrize("refresh", [False, True])
    def test_refresh_direction_flag(self, harmonic, positions, types, refresh):
        """Stored direction is perturbed by default, redrawn with the flag."""
        d0 = jnp.ones_like(positions) / jnp.sqrt(positions.size)
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        state = state.set(direction=d0)
        step = gmc_build_kernel(
            harmonic, n_reflect=3, refresh_direction=refresh
        )

        # Loose constraint: no reflections, every trajectory accepted, so the
        # final direction is the (perturbed or fresh) initial one.
        keys = jax.random.split(jax.random.key(0), 200)
        new_states, info = jax.jit(
            jax.vmap(lambda k: step(k, state, likelihood_constraint=1e6))
        )(keys)
        assert bool(jnp.all(info.accepted))

        cos = jnp.abs(jnp.sum(new_states.direction * d0, axis=(1, 2)))
        if refresh:
            # Random unit vectors in 12 dims: E|cos| ~ 0.23 (measured 0.23).
            assert float(cos.mean()) < 0.4
        else:
            # A 0.1 rad-scale nudge of d0 (measured mean 0.95).
            assert float(cos.mean()) > 0.9

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

    def test_acceptance_rate_reasonable(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = jax.jit(gmc_build_kernel(backend, n_reflect=5))

        n_steps = 100
        n_accepted = 0
        key = jax.random.key(123)

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            state, info = step(subkey, state, likelihood_constraint=2.0)
            n_accepted += int(info.accepted)

        rate = n_accepted / n_steps
        assert 0.1 < rate <= 1.0

    def test_scan_compatible(self, harmonic, positions, types):
        backend = harmonic
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = jax.jit(gmc_build_kernel(backend, n_reflect=3))

        def scan_step(carry, key):
            state = carry
            new_state, info = step(key, state, 2.0)
            return new_state, info.accepted

        keys = jax.random.split(jax.random.key(0), 10)
        final_state, accepted = jax.lax.scan(scan_step, state, keys)
        assert accepted.shape == (10,)

    def test_without_forces(self, harmonic, positions, types):
        """use_forces=False reflects in a random direction."""
        backend = harmonic
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = gmc_build_kernel(backend, n_reflect=3, use_forces=False)

        key = jax.random.key(0)
        new_state, info = jax.jit(step)(key, state, 10.0)

        assert info.n_evaluations == 3
        assert info.n_grad_evaluations == 0


class TestGalileanBaldockSemantics:
    """Check the trajectory semantics of the Galilean kernel.

    The position advances through violated regions, acceptance depends only
    on the final energy, a NaN energy triggers a reflection, and a rejection
    flips the perturbed direction.
    """

    def test_reject_when_no_path_below_constraint(
        self, harmonic, positions, types
    ):
        """With Emax below the initial energy the trajectory is rejected.

        The harmonic energy only grows away from the origin, so every
        reflection step lands above Emax.
        """
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.5
        )
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=4))

        new_state, info = step(
            jax.random.key(0), state, likelihood_constraint=0.1
        )

        assert not bool(info.accepted)
        assert jnp.allclose(new_state.positions, state.positions)

    def test_reject_flips_the_perturbed_direction(
        self, harmonic, positions, types
    ):
        """On reject the stored direction is minus the perturbed one.

        Flipping the stored direction instead would discard the perturbation
        the accept/reject decision was made for.
        """
        d0 = jnp.ones_like(positions) / jnp.sqrt(positions.size)
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.5
        )
        state = state.set(direction=d0)
        key = jax.random.key(0)
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=4))
        new_state, info = step(key, state, likelihood_constraint=0.1)
        assert not bool(info.accepted)

        # Same key split as the kernel: (init, perturb, reflect).
        _, key_perturb, _ = jax.random.split(key, 3)
        used = _perturb_direction(key_perturb, d0, 0.1)
        assert jnp.allclose(new_state.direction, -used, atol=1e-6)
        assert not jnp.allclose(new_state.direction, -d0, atol=1e-3)

    def test_persistent_direction_keeps_uniform_ball_invariant(self, harmonic):
        """A uniform ensemble stays uniform under repeated GMC calls.

        Start uniform in {0.5|q|^2 < 0.5} (a 6-D unit ball) with uniform
        directions and apply 100 persistent GMC calls at a fixed constraint.
        <r^2> must stay d/(d+2) = 0.75.  Flipping the unperturbed direction
        on reject drifts it inward to about 0.732.
        """
        n, n_atoms, d = 8000, 2, 6
        k_dir, k_rad, k_v, k_run = jax.random.split(jax.random.key(0), 4)
        x = jax.random.normal(k_dir, (n, d))
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        x = x * jax.random.uniform(k_rad, (n, 1)) ** (1 / d)
        x = x.reshape(n, n_atoms, 3)
        v = jax.random.normal(k_v, x.shape)
        v = v / jnp.linalg.norm(v.reshape(n, -1), axis=1)[:, None, None]

        state = jax.vmap(
            lambda p, e: _make_gmc_state(
                p, jnp.zeros(n_atoms, jnp.int32), e, step_size=0.15
            )
        )(x, 0.5 * jnp.sum(x**2, axis=(1, 2)))
        state = state.set(direction=v)
        step = jax.vmap(
            gmc_build_kernel(harmonic, n_reflect=4), in_axes=(0, 0, None)
        )

        def body(st, key):
            st, _ = step(jax.random.split(key, n), st, 0.5)
            return st, None

        final, _ = jax.jit(
            lambda st: jax.lax.scan(body, st, jax.random.split(k_run, 100))
        )(state)
        r2 = jnp.sum(final.positions**2, axis=(1, 2))
        # statistical error ~0.0022; the old kernel was ~0.018 low
        assert abs(float(r2.mean()) - d / (d + 2)) < 0.006, float(r2.mean())

    def test_accept_advances_through_intermediate_violations(
        self, harmonic, positions, types
    ):
        """Accepted moves change the positions, and acceptance is not rare."""
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.05
        )
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=5))
        key = jax.random.key(7)
        accept_count = 0
        for _ in range(50):
            key, sub = jax.random.split(key)
            new_state, info = step(sub, state, likelihood_constraint=0.5)
            if bool(info.accepted):
                accept_count += 1
                assert not jnp.allclose(new_state.positions, state.positions)
            state = new_state
        assert accept_count > 5

    def test_nan_energy_triggers_reflection_not_silent_reject(
        self, positions, types
    ):
        """A NaN energy is treated as a violation, like an energy above Emax.

        The backend returns NaN at any position other than the initial one.
        Every step violates, so the final energy is +inf and the move is
        rejected with the state intact.
        """

        def nan_backend(p, types, cell, max_neighbors=0, ensemble_params=None):
            moved = jnp.any(jnp.abs(p - positions) > 1e-6)
            return BackendResult(energy=jnp.where(moved, jnp.nan, 0.25))

        state = _make_gmc_state(positions, types, energy=0.25, step_size=0.05)
        step = jax.jit(
            gmc_build_kernel(nan_backend, n_reflect=3, use_forces=False)
        )

        new_state, info = step(
            jax.random.key(0), state, likelihood_constraint=10.0
        )

        assert not bool(info.accepted)
        assert jnp.allclose(new_state.positions, state.positions)
        assert float(new_state.energy) == 0.25

    def test_final_energy_below_constraint_when_accepted(
        self, harmonic, positions, types
    ):
        """Every accepted move leaves state.energy below Emax."""
        state = _make_gmc_state(
            positions, types, energy=_energy(positions), step_size=0.02
        )
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=5))
        key = jax.random.key(99)
        n_accepted = 0
        for _ in range(60):
            key, sub = jax.random.split(key)
            state, info = step(sub, state, likelihood_constraint=1.0)
            if bool(info.accepted):
                n_accepted += 1
                assert float(state.energy) < 1.0
        assert n_accepted > 0


class TestGalileanSpeciesMask:
    """Check that the kernel moves only the atoms of the scoped species.

    The tests cover the frozen atoms in the accept, reject and reflection
    paths, the direction confined to the moving atoms, and a mask that
    follows state.types.
    """

    @pytest.mark.parametrize(
        "constraint",
        [
            10.0,  # loose: trajectory stays in-region, move accepts
            0.55,  # tight: forces reflections and rejections
        ],
    )
    @pytest.mark.parametrize("use_forces", [True, False])
    def test_frozen_species_never_moves(
        self, harmonic, positions, types, constraint, use_forces
    ):
        """The type-1 atoms are bit-identical after many steps.

        The two constraints exercise the accept and the reject path.
        ``use_forces=False`` covers the random-reflection branch, which
        applies the mask separately.
        """
        state = _make_gmc_state(
            positions, types, _energy(positions), step_size=0.2
        )
        step = jax.jit(
            gmc_build_kernel(
                harmonic,
                n_reflect=3,
                species=(0,),
                use_forces=use_forces,
            )
        )

        initial = state.positions
        for i in range(20):
            state, _ = step(jax.random.key(i), state, constraint)

        # Frozen sublattice: exactly unchanged, not merely close.
        assert jnp.array_equal(state.positions[2:], initial[2:])

    def test_moving_species_actually_moves(self, harmonic, positions, types):
        state = _make_gmc_state(positions, types, _energy(positions))
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=(0,)))

        initial = state.positions
        for i in range(10):
            state, _ = step(jax.random.key(i), state, 10.0)

        assert not jnp.allclose(state.positions[:2], initial[:2])

    def test_direction_is_confined_to_the_subspace(
        self, harmonic, positions, types
    ):
        """Frozen rows of the direction stay zero; the whole has unit norm."""
        state = _make_gmc_state(positions, types, _energy(positions))
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=(0,)))

        for i in range(5):
            state, _ = step(jax.random.key(i), state, 10.0)
            assert jnp.array_equal(state.direction[2:], jnp.zeros((2, 3)))
            norm = jnp.sqrt(jnp.sum(state.direction**2))
            assert jnp.isclose(norm, 1.0, atol=1e-5)

    def test_multiple_species_in_one_scope(self, harmonic, positions, types):
        """species=(0, 1) on a 2-species system equals the unscoped move.

        The two kernels give bit-identical results for the same key.
        """
        state = _make_gmc_state(positions, types, _energy(positions))
        scoped = jax.jit(
            gmc_build_kernel(harmonic, n_reflect=3, species=(0, 1))
        )
        unscoped = jax.jit(gmc_build_kernel(harmonic, n_reflect=3))

        key = jax.random.key(7)
        a, info_a = scoped(key, state, 10.0)
        b, info_b = unscoped(key, state, 10.0)

        assert jnp.array_equal(a.positions, b.positions)
        assert jnp.array_equal(a.direction, b.direction)
        assert jnp.array_equal(a.energy, b.energy)
        assert bool(info_a.accepted) == bool(info_b.accepted)

    def test_custom_direction_field(self, harmonic, positions, types):
        state = _make_gmc_state(
            positions,
            types,
            _energy(positions),
            direction_field="direction_gmc_Ge",
        )
        step = jax.jit(
            gmc_build_kernel(
                harmonic,
                n_reflect=3,
                species=(0,),
                direction_field="direction_gmc_Ge",
            )
        )
        new_state, _ = step(jax.random.key(0), state, 10.0)

        norm = jnp.sqrt(jnp.sum(new_state.direction_gmc_Ge**2))
        assert norm > 0.1

    def test_mask_follows_types_not_positions(
        self, harmonic, positions, types
    ):
        """The mask is read from ``state.types``, so a swap retargets the move.

        A mask fixed at build time would keep moving the original atoms after
        a swap changes the types.
        """
        step = jax.jit(gmc_build_kernel(harmonic, n_reflect=3, species=(0,)))

        # Reverse the species assignment: now atoms 2, 3 are the movers.
        state = _make_gmc_state(
            positions,
            jnp.array([1, 1, 0, 0]),
            _energy(positions),
            step_size=0.2,
        )
        initial = state.positions
        for i in range(10):
            state, _ = step(jax.random.key(i), state, 10.0)

        assert jnp.array_equal(state.positions[:2], initial[:2])
        assert not jnp.allclose(state.positions[2:], initial[2:])


class TestScopedMovesInMWG:
    """Check that two scoped GMC moves work side by side in build_mwg."""

    def _descriptors(self):
        ge = GMCMoveSpec(species=["Ge"], step_size=0.3, n_reflect=3)
        si = GMCMoveSpec(species=["Si"], step_size=0.05, n_reflect=3)
        return [
            ge.to_descriptor(symbol_map=_SYMBOL_MAP),
            si.to_descriptor(symbol_map=_SYMBOL_MAP),
        ]

    def _init_state(self, harmonic, positions, types):
        init_fn, step_fn, per_move_fns = build_mwg(
            harmonic, self._descriptors()
        )
        state = init_fn(
            positions,
            types,
            energy=0.5 * jnp.sum(positions**2),
            cell=jnp.zeros((3, 3)),
        )
        return state, step_fn, per_move_fns

    def test_separate_direction_fields_and_step_sizes(
        self, harmonic, positions, types
    ):
        descs = self._descriptors()
        assert [d.name for d in descs] == ["gmc_Ge", "gmc_Si"]

        state, _, _ = self._init_state(harmonic, positions, types)

        # build_mwg must keep the two direction fields distinct.
        assert hasattr(state, "direction_gmc_Ge")
        assert hasattr(state, "direction_gmc_Si")
        # Per-move step sizes, seeded from each spec.
        assert state.step_sizes.shape == (2,)
        assert jnp.allclose(state.step_sizes, jnp.array([0.3, 0.05]))

    def test_each_move_only_touches_its_own_sublattice(
        self, harmonic, positions, types
    ):
        state, _, per_move_fns = self._init_state(harmonic, positions, types)
        ge_move, si_move = per_move_fns

        after_ge = ge_move(state, jax.random.key(1), 10.0)[0]
        assert jnp.array_equal(after_ge.positions[2:], state.positions[2:])
        # The Si move's direction is untouched by the Ge move.
        assert jnp.array_equal(
            after_ge.direction_gmc_Si, state.direction_gmc_Si
        )

        after_si = si_move(after_ge, jax.random.key(2), 10.0)[0]
        assert jnp.array_equal(after_si.positions[:2], after_ge.positions[:2])
        # The Ge direction also survives the Si move.
        assert jnp.array_equal(
            after_si.direction_gmc_Ge, after_ge.direction_gmc_Ge
        )

    def test_both_sublattices_move_under_the_full_mwg_step(
        self, harmonic, positions, types
    ):
        state, step_fn, _ = self._init_state(harmonic, positions, types)
        initial = state.positions

        jitted = jax.jit(step_fn)
        for i in range(40):
            state, _ = jitted(
                jax.random.key(i), state, 10.0, jax.random.key(1000 + i)
            )

        assert not jnp.allclose(state.positions[:2], initial[:2])
        assert not jnp.allclose(state.positions[2:], initial[2:])
        assert int(jnp.sum(state.n_proposed)) == 40
