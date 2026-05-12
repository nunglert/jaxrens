"""NeuralIL backend wrapper.

Wraps NeuralIL behind the EnergyBackend interface. Supports both single
``NeuralIL`` / ``NeuralILwithMorse`` models and ``PlainEnsemble`` /
``PlainEnsemblewithMorse`` ensembles; the layout is auto-detected from
the saved params at load time. NeuralIL computes neighbors implicitly
via supercell expansion during descriptor calculation — no explicit
neighbor list. The dynamics model is built **once** at construction
time and reused; ``max_neighbors`` is passed at call time as a
buffer-shape argument (mirroring the MACE backend's design). Different
``max_neighbors`` values trigger a JIT retrace because of the static
shape, but no Python-side model rebuild.

If the saved model carries a per-atom energy baseline (under
``model_info.specific_info["energy_shift_per_atom"]``), the backend
adds ``shift * n_real_atoms`` back to the predicted total energy so
calling code sees absolute energies. Training scripts subtract this
shift from targets for numerical stability.

NeuralIL is an optional dependency. All imports are guarded.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Sequence

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded NeuralIL imports
# ---------------------------------------------------------------------------

_NEURALIL_AVAILABLE = False
_NEURALIL_IMPORT_ERROR: str | None = None

try:
    from neuralil.bessel_descriptors import (
        PowerSpectrumGenerator,
        _get_max_number_of_neighbors,
    )
    from neuralil.model import NeuralIL, NeuralILwithMorse, ResNetCore
    from neuralil.plain_ensembles.model import (
        PlainEnsemble,
        PlainEnsemblewithMorse,
    )
    from neuralil.plain_ensembles.training import get_n_models

    _NEURALIL_AVAILABLE = True
except ImportError as exc:
    _NEURALIL_IMPORT_ERROR = str(exc)


def _require_neuralil() -> None:
    if not _NEURALIL_AVAILABLE:
        raise ImportError(
            "NeuralIL is required for the neuralil backend but is not installed. "
            f"Original import error: {_NEURALIL_IMPORT_ERROR}\n"
            "Install it with: pip install neuralil"
        )


def is_available() -> bool:
    return _NEURALIL_AVAILABLE


# ---------------------------------------------------------------------------
# Params-layout detection
# ---------------------------------------------------------------------------


def _detect_layout(params: dict) -> tuple[bool, bool]:
    """Return ``(is_ensemble, has_morse)`` from the saved Flax params dict.

    A ``PlainEnsemble[withMorse]`` nests its sub-model under a
    ``"neuralil"`` (or ``"neuralil_w_morse"``) key inside ``params``; a
    single ``NeuralIL[withMorse]`` does not, exposing its submodules
    (``core_model``, ``embed``, ``denormalizer``, optionally ``morse``)
    directly.
    """
    inner = params.get("params", params)
    ensemble_keys = {"neuralil", "neuralil_w_morse"}
    ensemble_match = ensemble_keys & inner.keys()
    if ensemble_match:
        sub = inner[next(iter(ensemble_match))]
        return True, "morse" in sub
    return False, "morse" in inner


# ---------------------------------------------------------------------------
# NeuralILBackend
# ---------------------------------------------------------------------------


def _build_dynamics_model(
    n_types: int,
    embed_d: int,
    r_cutoff: float,
    n_max: int,
    core_widths: list[int],
    supercell_trafo: tuple[int, int, int],
    has_morse: bool,
    is_ensemble: bool,
    n_ensemble: int,
):
    """Build the (max_neighbors-independent) NeuralIL dynamics model.

    Returns a single ``NeuralIL[withMorse]`` when ``is_ensemble`` is
    False, otherwise the corresponding ``PlainEnsemble[withMorse]``
    wrapper.
    """
    descriptor_gen = PowerSpectrumGenerator(
        n_max, r_cutoff, n_types, supercell_trafo,
    )
    core_model = ResNetCore(core_widths)

    if has_morse:
        # neuralil >= some-version replaced ``morse_type`` with a flax
        # ``mixer`` submodule.  Constructing with defaults reproduces
        # the layout the pickle was trained against (since the pickle
        # encodes whatever module tree the installed neuralil exposes).
        individual = NeuralILwithMorse(
            n_types, embed_d, r_cutoff,
            descriptor_gen, descriptor_gen.process_some_data,
            core_model,
        )
    else:
        individual = NeuralIL(
            n_types, embed_d, r_cutoff,
            descriptor_gen, descriptor_gen.process_some_data,
            core_model,
        )

    if not is_ensemble:
        return individual

    if has_morse:
        return PlainEnsemblewithMorse(individual, n_ensemble)
    return PlainEnsemble(individual, n_ensemble)


class NeuralILBackend:
    """NeuralIL energy backend for jaxrens.

    The dynamics model is built once at construction time. ``max_neighbors``
    is a runtime call-time argument (static at trace time) that shapes
    the neighbor-buffer slice inside the descriptor generator.

    Single-model and ensemble checkpoints are both supported; the layout
    is selected via ``is_ensemble`` and the energy is reduced
    accordingly (``.mean()`` over the ensemble axis, identity for a
    single model).
    """

    def __init__(
        self,
        model_params: dict,
        r_cutoff: float,
        sorted_elements: list[str],
        supercell_trafo: tuple[int, int, int],
        n_max: int,
        embed_d: int,
        core_widths: list[int],
        is_ensemble: bool,
        has_morse: bool,
        n_ensemble: int = 1,
        energy_shift_per_atom: float = 0.0,
    ):
        self.r_cutoff = r_cutoff
        self.model_params = model_params
        self.sorted_elements = sorted_elements
        self.supercell_trafo = supercell_trafo
        self.n_max = n_max
        self.embed_d = embed_d
        self.core_widths = core_widths
        self.is_ensemble = is_ensemble
        self.n_ensemble = n_ensemble if is_ensemble else 1
        self.has_morse = has_morse
        self.energy_shift_per_atom = float(energy_shift_per_atom)
        self._dynamics_model = _build_dynamics_model(
            n_types=len(sorted_elements),
            embed_d=embed_d,
            r_cutoff=r_cutoff,
            n_max=n_max,
            core_widths=core_widths,
            supercell_trafo=supercell_trafo,
            has_morse=has_morse,
            is_ensemble=is_ensemble,
            n_ensemble=self.n_ensemble,
        )

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        """Atomic numbers (Z) of the elements the model was trained on,
        in the same order as ``sorted_elements``. Used by the resolver to
        map user-supplied Z numbers in ``start_species`` to the model's
        contiguous 0-based type indices."""
        from ase.data import atomic_numbers as _Z

        return tuple(_Z[s] for s in self.sorted_elements)

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        # Non-periodic: use large dummy cell
        safe_cell = jnp.where(
            jnp.trace(cell) == 0, 1000.0 * jnp.eye(3), cell,
        )

        # Compute energy. max_neighbors is a static-at-trace int that flows
        # down into PSG._process_center as a buffer-shape arg. For an
        # ensemble the output is shape (n_ensemble,) → reduce by mean; for
        # a single model the output is scalar and .mean() is identity.
        energy_out = self._dynamics_model.apply(
            self.model_params,
            positions,
            species,
            safe_cell,
            max_neighbors,
            method=self._dynamics_model.calc_potential_energy,
        )
        energy = energy_out.mean() if self.is_ensemble else energy_out

        # Re-add the per-atom baseline that training subtracted from targets.
        if self.energy_shift_per_atom != 0.0:
            n_real = (species >= 0).sum()
            energy = energy + self.energy_shift_per_atom * n_real

        # Check actual neighbor count for overflow detection
        sc_a, sc_b, sc_c = self.supercell_trafo
        actual_max_neighbors = _get_max_number_of_neighbors(
            positions, species, self.r_cutoff, safe_cell, sc_a, sc_b, sc_c,
        )
        overflow = actual_max_neighbors > max_neighbors

        return energy, actual_max_neighbors, overflow

    def max_neighbors_for(
        self,
        positions: jnp.ndarray,
        cell: jnp.ndarray,
    ) -> jnp.ndarray:
        """Geometry-only per-walker max neighbor count.

        Mirrors MACE's same-named method (``mace.py:224-237``) so the
        resolver's ``_finalise_initial_energies_and_counts`` can take
        the bucket-aware branch instead of falling through to the
        ``max_neighbors=0`` path.  No NeuralIL forward pass is
        triggered — the result is a static-shape buffer sizing hint
        derived purely from coordinates + the model's r_cutoff +
        supercell_trafo.

        Pass dummy zero ``species`` since neighbor counting depends
        only on positions and the cutoff (species enter only through
        a padded-atom mask which ``_get_max_number_of_neighbors``
        ignores for length-N inputs).
        """
        sc_a, sc_b, sc_c = self.supercell_trafo
        safe_cell = jnp.where(
            jnp.trace(cell) == 0, 1000.0 * jnp.eye(3), cell,
        )
        species = jnp.zeros(positions.shape[0], dtype=jnp.int32)
        return _get_max_number_of_neighbors(
            positions, species, self.r_cutoff, safe_cell, sc_a, sc_b, sc_c,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_neuralil(
    pickle_file: str | None = None,
    supercell_trafo: Sequence[int] = (1, 1, 1),
    **kwargs: Any,
) -> NeuralILBackend:
    """Create a NeuralIL energy backend.

    Auto-detects whether the pickle is a single ``NeuralIL[withMorse]``
    or a ``PlainEnsemble[withMorse]`` from the saved params layout, and
    builds the matching dynamics model.

    Args:
        pickle_file: Path to NeuralIL model pickle (``NeuralILModelInfo``).
            Either single-model or ensemble checkpoints are accepted.
        supercell_trafo: Supercell diagonal transformation (s_a, s_b, s_c).

    Returns:
        NeuralILBackend instance.
    """
    _require_neuralil()

    if pickle_file is None:
        raise ValueError(
            "pickle_file is required for the neuralil backend. "
            "Pass the path to your NeuralIL model pickle."
        )

    with open(pickle_file, "rb") as f:
        model_info = pickle.load(f)

    is_ensemble, has_morse = _detect_layout(model_info.params)
    n_ensemble = get_n_models(model_info.params) if is_ensemble else 1

    energy_shift_per_atom = 0.0
    specific_info = getattr(model_info, "specific_info", None)
    if isinstance(specific_info, dict):
        energy_shift_per_atom = float(
            specific_info.get("energy_shift_per_atom", 0.0)
        )

    logger.info(
        "NeuralIL backend created: r_cut=%.2f, elements=%s, supercell=%s, "
        "ensemble=%s, n_ensemble=%d, morse=%s, energy_shift_per_atom=%g",
        model_info.r_cut, model_info.sorted_elements, supercell_trafo,
        is_ensemble, n_ensemble, has_morse, energy_shift_per_atom,
    )

    return NeuralILBackend(
        model_params=model_info.params,
        r_cutoff=model_info.r_cut,
        sorted_elements=model_info.sorted_elements,
        supercell_trafo=tuple(supercell_trafo),
        n_max=model_info.n_max,
        embed_d=model_info.embed_d,
        core_widths=model_info.core_widths,
        is_ensemble=is_ensemble,
        n_ensemble=n_ensemble,
        has_morse=has_morse,
        energy_shift_per_atom=energy_shift_per_atom,
    )
