"""GMC ``use_forces`` in the YAML schema: random reflection as a baseline."""

import jax
import jax.numpy as jnp

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.schema.moves import GMCMoveSpec
from jaxrens.sampling.moves.galilean import build_kernel
from jaxrens.state.mc_state import make_mc_state_class

_POSITIONS = jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])


def test_default_spec_keeps_the_historical_kernel_kwargs():
    assert GMCMoveSpec(n_reflect=3)._kernel_kwargs() == {"n_reflect": 3}
    assert GMCMoveSpec().use_forces is True


def test_use_forces_false_reaches_the_kernel():
    kwargs = GMCMoveSpec(n_reflect=3, use_forces=False)._kernel_kwargs()
    assert kwargs == {"n_reflect": 3, "use_forces": False}


def test_random_reflection_evaluates_no_forces():
    State = make_mc_state_class({"direction": jnp.ndarray})
    state = State(
        positions=_POSITIONS,
        types=jnp.zeros(3, dtype=jnp.int32),
        energy=jnp.asarray(0.5 * jnp.sum(_POSITIONS**2)),
        cell=jnp.zeros((3, 3)),
        step_size=jnp.asarray(0.1),
        step_sizes=jnp.full(1, 0.1),
        n_accepted=jnp.zeros(1, dtype=jnp.int32),
        n_proposed=jnp.zeros(1, dtype=jnp.int32),
        max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
        overflow=jnp.asarray(False),
        ensemble_params={},
        direction=jnp.zeros_like(_POSITIONS),
    )
    kernel = build_kernel(create_harmonic(k=1.0), **GMCMoveSpec(n_reflect=4, use_forces=False)._kernel_kwargs())
    new, info = jax.jit(kernel)(jax.random.key(0), state, jnp.asarray(0.6))
    assert int(info.n_grad_evaluations) == 0
    assert int(info.n_evaluations) == 4
    assert bool(info.accepted) == bool(new.energy < 0.6)
