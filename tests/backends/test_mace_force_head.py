"""MACEBackend force_source: head (direct, non-conservative) vs autograd forces.

The head model comes from mace-nonconservative (the torch fork with the
non-conservative force head) and is converted with mace-jax's torch->JAX
converter inside the test, so both the torch fork and a mace-jax with the head
must be importable; otherwise the tests skip.
"""

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.mace import MACEBackend, is_available

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

pytestmark = pytest.mark.mace


def _head_fork_available() -> bool:
    if not is_available() or importlib.util.find_spec("mace") is None:
        return False
    from mace.modules import blocks

    return hasattr(blocks, "NonLinearForcesReadoutBlock")


head_required = pytest.mark.skipif(
    not _head_fork_available(),
    reason="needs mace-nonconservative (torch) and mace-jax with the force head",
)

FIXTURE_DIR = Path(__file__).parent.parent / "_assets" / "models" / "mace_mp_small"
R_MAX = 4.0

# 4-atom Si cell: 2 species-0 atoms of the model below are enough
POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [1.4, 1.3, 1.5], [2.7, 2.6, 0.1], [0.2, 2.8, 2.6]],
    dtype=np.float32,
)
CELL = np.eye(3, dtype=np.float32) * 5.6


@pytest.fixture(scope="module")
def head_model():
    """A small torch MACE with the force head, its JAX conversion, and torch outputs."""
    import torch
    from e3nn import o3
    from mace import data, modules, tools
    from mace.tools import torch_geometric
    from mace.tools.scripts_utils import extract_config_mace_model
    from mace_jax.cli.mace_jax_from_torch import convert_model

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    model = modules.ScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e + 8x1o"),
        atomic_energies=np.array([-5.0]),
        avg_num_neighbors=8.0,
        atomic_numbers=[14],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    ).eval()
    with torch.no_grad():  # a fresh head predicts ~1e-4; make the check sensitive
        for p in model.force_readout.parameters():
            p.mul_(10.0)

    config = data.Configuration(
        atomic_numbers=np.full(4, 14),
        positions=POSITIONS.astype(float),
        cell=CELL.astype(float),
        pbc=(True, True, True),
        properties={},
        property_weights={},
    )
    atomic = data.AtomicData.from_config(
        config, z_table=tools.AtomicNumberTable([14]), cutoff=R_MAX, heads=["Default"]
    )
    batch = next(iter(torch_geometric.dataloader.DataLoader([atomic], batch_size=1)))
    out = model(batch.to_dict(), training=False, evaluating=True, compute_force=True)

    cfg = extract_config_mace_model(model)
    cfg["torch_model_class"] = type(model).__name__
    graphdef, state, _ = convert_model(model, cfg)
    return {
        "graphdef": graphdef,
        "state": state,
        "torch_energy": float(out["energy"].detach()[0]),
        "torch_head": out["forces"].detach().numpy(),
        "torch_autograd": out["backpropagated_forces"].detach().numpy(),
    }


def _backend(model, force_source):
    return MACEBackend(
        model["graphdef"], model["state"], R_MAX, 1, [14],
        supercell_trafo=(3, 3, 3), force_source=force_source,
    )


def _evaluate(backend):
    species = jnp.zeros(4, dtype=jnp.int32)
    fn = jax.jit(lambda p, c: backend.energy_and_forces(p, species, c, 64))
    return fn(jnp.asarray(POSITIONS), jnp.asarray(CELL))


@head_required
class TestForceSource:
    def test_head_forces_are_the_models_head_forces(self, head_model):
        with jax.default_matmul_precision("highest"):
            res = _evaluate(_backend(head_model, "head"))
        assert not bool(res.overflow)
        np.testing.assert_allclose(np.asarray(res.forces), head_model["torch_head"], atol=1e-4)
        assert float(res.energy) == pytest.approx(head_model["torch_energy"], abs=1e-4)
        # the head is not the energy gradient: the two force sets differ
        assert np.abs(head_model["torch_head"] - head_model["torch_autograd"]).max() > 1e-3

    def test_autograd_forces_are_minus_the_energy_gradient(self, head_model):
        with jax.default_matmul_precision("highest"):
            res = _evaluate(_backend(head_model, "autograd"))
        np.testing.assert_allclose(np.asarray(res.forces), head_model["torch_autograd"], atol=1e-4)

    def test_energy_does_not_depend_on_the_force_source(self, head_model):
        species = jnp.zeros(4, dtype=jnp.int32)
        pos, cell = jnp.asarray(POSITIONS), jnp.asarray(CELL)
        e_head = _backend(head_model, "head")(pos, species, cell, 64).energy
        e_auto = _backend(head_model, "autograd")(pos, species, cell, 64).energy
        assert float(e_head) == float(e_auto)

    def test_unknown_force_source_raises(self, head_model):
        with pytest.raises(ValueError, match="force_source"):
            _backend(head_model, "magic")


@pytest.mark.skipif(not is_available(), reason="mace-jax not installed")
def test_head_needs_a_model_with_a_head():
    from jaxrens.backends.mace import create_mace

    with pytest.raises(ValueError, match="force head"):
        create_mace(model_path=str(FIXTURE_DIR), supercell_trafo=(4, 4, 4), force_source="head")


def test_schema_passes_force_source(monkeypatch):
    from jaxrens.cli.schema.backend import MACEBackendSpec

    seen = {}

    def fake_create_mace(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr("jaxrens.backends.mace.create_mace", fake_create_mace)
    MACEBackendSpec(checkpoint_path="m.npz", force_source="head").build_backend()
    assert seen["force_source"] == "head"
    assert MACEBackendSpec(checkpoint_path="m.npz").force_source == "autograd"
