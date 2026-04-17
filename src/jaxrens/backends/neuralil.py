"""NeuralIL backend wrapper.

Wraps NeuralIL's PlainEnsemble behind the EnergyBackend interface.
NeuralIL computes neighbors implicitly via supercell expansion during
descriptor calculation — no explicit neighbor list. The `max_neighbors`
parameter is a compile-time constant baked into the descriptor generator;
JAX's compilation cache handles retrace when the value changes.

When `max_neighbors` changes (e.g. after overflow in the outer NS loop),
the backend lazily builds a new model instance with the updated value
and caches it. Since `max_neighbors` is a static argument, JAX retraces
`__call__` in Python before compiling, so the Python-level model
construction happens outside the JIT boundary.

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
# NeuralILBackend
# ---------------------------------------------------------------------------


def _build_dynamics_model(
    n_types: int,
    embed_d: int,
    r_cutoff: float,
    n_max: int,
    core_widths: list[int],
    max_neighbors: int,
    supercell_trafo: tuple[int, int, int],
    has_morse: bool,
    n_ensemble: int,
):
    """Build a NeuralIL dynamics model for a specific max_neighbors value."""
    descriptor_gen = PowerSpectrumGenerator(
        n_max, r_cutoff, n_types, max_neighbors, supercell_trafo,
    )
    core_model = ResNetCore(core_widths)

    if has_morse:
        individual = NeuralILwithMorse(
            n_types, embed_d, r_cutoff,
            descriptor_gen, descriptor_gen.process_some_data,
            core_model, morse_type="RepulsiveMorse",
        )
        return PlainEnsemblewithMorse(individual, n_ensemble)
    else:
        individual = NeuralIL(
            n_types, embed_d, r_cutoff,
            descriptor_gen, descriptor_gen.process_some_data,
            core_model,
        )
        return PlainEnsemble(individual, n_ensemble)


class NeuralILBackend:
    """NeuralIL energy backend for jaxrens.

    Model params are stored on the instance. The dynamics model is
    lazily built and cached per `max_neighbors` value.

    Since `max_neighbors` is a static argument in jaxrens (static_field
    on MCState), JAX retraces when it changes. During retrace, Python
    executes `__call__` which triggers the lazy model build — this
    happens outside the JIT boundary, so Python object construction
    is safe.
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
        n_ensemble: int,
        has_morse: bool,
    ):
        self.r_cutoff = r_cutoff
        self.model_params = model_params
        self.sorted_elements = sorted_elements
        self.supercell_trafo = supercell_trafo
        self.n_max = n_max
        self.embed_d = embed_d
        self.core_widths = core_widths
        self.n_ensemble = n_ensemble
        self.has_morse = has_morse
        self._model_cache: dict[int, Any] = {}

    def _get_model(self, max_neighbors: int):
        """Get or build a dynamics model for this max_neighbors value."""
        if max_neighbors not in self._model_cache:
            logger.info("Building NeuralIL model for max_neighbors=%d", max_neighbors)
            self._model_cache[max_neighbors] = _build_dynamics_model(
                n_types=len(self.sorted_elements),
                embed_d=self.embed_d,
                r_cutoff=self.r_cutoff,
                n_max=self.n_max,
                core_widths=self.core_widths,
                max_neighbors=max_neighbors,
                supercell_trafo=self.supercell_trafo,
                has_morse=self.has_morse,
                n_ensemble=self.n_ensemble,
            )
        return self._model_cache[max_neighbors]

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        # Get cached model for this max_neighbors (built lazily)
        dynamics_model = self._get_model(max_neighbors)

        # Non-periodic: use large dummy cell
        safe_cell = jnp.where(
            jnp.trace(cell) == 0, 1000.0 * jnp.eye(3), cell,
        )

        # Compute energy (ensemble mean)
        energy_ensemble = dynamics_model.apply(
            self.model_params,
            positions,
            species,
            safe_cell,
            method=dynamics_model.calc_potential_energy,
        )
        energy = energy_ensemble.mean()

        # Check actual neighbor count for overflow detection
        sc_a, sc_b, sc_c = self.supercell_trafo
        actual_max_neighbors = _get_max_number_of_neighbors(
            positions, species, self.r_cutoff, safe_cell, sc_a, sc_b, sc_c,
        )
        overflow = actual_max_neighbors > max_neighbors

        return energy, actual_max_neighbors, overflow


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_neuralil(
    pickle_file: str | None = None,
    supercell_trafo: Sequence[int] = (1, 1, 1),
    **kwargs: Any,
) -> NeuralILBackend:
    """Create a NeuralIL energy backend.

    Args:
        pickle_file: Path to NeuralIL model pickle (NeuralILModelInfo).
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

    has_morse = "morse" in model_info.params["params"]["neuralil"]
    n_ensemble = get_n_models(model_info.params)

    logger.info(
        "NeuralIL backend created: r_cut=%.2f, elements=%s, supercell=%s, "
        "n_ensemble=%d, morse=%s",
        model_info.r_cut, model_info.sorted_elements, supercell_trafo,
        n_ensemble, has_morse,
    )

    return NeuralILBackend(
        model_params=model_info.params,
        r_cutoff=model_info.r_cut,
        sorted_elements=model_info.sorted_elements,
        supercell_trafo=tuple(supercell_trafo),
        n_max=model_info.n_max,
        embed_d=model_info.embed_d,
        core_widths=model_info.core_widths,
        n_ensemble=n_ensemble,
        has_morse=has_morse,
    )
