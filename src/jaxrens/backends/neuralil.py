"""NeuralIL backend wrapper.

Wraps NeuralIL's PlainEnsemble `calc_potential_energy(positions, types, cell)`
behind the EnergyFn protocol. NeuralIL computes neighbors implicitly during
descriptor calculation -- no neighbor_list argument. The `max_neighbors`
parameter is a compile-time constant baked into the descriptor generator.

For multi-kernel dispatch across different max_neighbors values, use
`create_neuralil_kernel_set()` which returns a `CompiledKernelSet`.

NeuralIL is an optional dependency. All imports are guarded with try/except
so that the rest of jaxrens works without it installed.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Sequence

import jax.numpy as jnp

from jaxrens.types import Box, Params, Positions, Types

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
    """Raise a clear error if NeuralIL is not installed."""
    if not _NEURALIL_AVAILABLE:
        raise ImportError(
            "NeuralIL is required for the neuralil backend but is not installed. "
            f"Original import error: {_NEURALIL_IMPORT_ERROR}\n"
            "Install it with: pip install neuralil"
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_neuralil_model(
    pickle_file: str,
    max_neighbors: int,
    supercell_trafo: Sequence[int],
) -> tuple[Any, Any]:
    """Load a NeuralIL PlainEnsemble model from a pickle file.

    Args:
        pickle_file: Path to the pickle file containing model_info.
        max_neighbors: Compile-time max neighbor count for descriptors.
        supercell_trafo: Diagonal supercell transformation (s_a, s_b, s_c).

    Returns:
        (model_info, dynamics_model) tuple.
    """
    _require_neuralil()

    with open(pickle_file, "rb") as f:
        model_info = pickle.load(f)

    descriptor_generator = PowerSpectrumGenerator(
        model_info.n_max,
        model_info.r_cut,
        len(model_info.sorted_elements),
        max_neighbors,
        supercell_trafo,
    )

    core_model = ResNetCore(model_info.core_widths)
    n_ensemble = get_n_models(model_info.params)

    if "morse" in model_info.params["params"]["neuralil"]:
        individual_model = NeuralILwithMorse(
            len(model_info.sorted_elements),
            model_info.embed_d,
            model_info.r_cut,
            descriptor_generator,
            descriptor_generator.process_some_data,
            core_model,
            morse_type="RepulsiveMorse",
        )
        dynamics_model = PlainEnsemblewithMorse(individual_model, n_ensemble)
    else:
        individual_model = NeuralIL(
            len(model_info.sorted_elements),
            model_info.embed_d,
            model_info.r_cut,
            descriptor_generator,
            descriptor_generator.process_some_data,
            core_model,
        )
        dynamics_model = PlainEnsemble(individual_model, n_ensemble)

    return model_info, dynamics_model


# ---------------------------------------------------------------------------
# Energy function factory (single max_neighbors)
# ---------------------------------------------------------------------------


def _make_energy_fn(
    pickle_file: str,
    max_neighbors: int,
    supercell_trafo: Sequence[int],
) -> tuple[Any, Params]:
    """Create an EnergyFn for a specific max_neighbors value.

    The returned function conforms to the EnergyFn protocol. Model params
    are baked into the closure (they are immutable NeuralIL weights), so
    the `params` dict returned here carries metadata only.

    Args:
        pickle_file: Path to NeuralIL model pickle.
        max_neighbors: Compile-time neighbor bound.
        supercell_trafo: Supercell diagonal.

    Returns:
        (energy_fn, params) tuple.
    """
    _require_neuralil()

    model_info, dynamics_model = load_neuralil_model(
        pickle_file, max_neighbors, supercell_trafo
    )

    s_a, s_b, s_c = supercell_trafo
    r_cut = model_info.r_cut
    model_params = model_info.params

    def energy_fn(
        params: Params,
        positions: Positions,
        types: Types,
        cell: Box | None = None,
        **unused_kwargs: Any,
    ) -> jnp.ndarray:
        """Compute potential energy via NeuralIL ensemble mean.

        If cell is None, a large dummy cell is used (non-periodic).
        """
        cell = cell if cell is not None else 1000.0 * jnp.eye(3)

        energy_ensemble = dynamics_model.apply(
            model_params,
            positions,
            types,
            cell,
            method=dynamics_model.calc_potential_energy,
        )
        return energy_ensemble.mean()

    params: Params = {
        "max_neighbors": max_neighbors,
        "r_cut": r_cut,
        "supercell_trafo": list(supercell_trafo),
        "pickle_file": pickle_file,
    }

    return energy_fn, params


# ---------------------------------------------------------------------------
# Violation-aware energy function (returns neighbors + violations)
# ---------------------------------------------------------------------------


def _make_checked_energy_fn(
    pickle_file: str,
    max_neighbors: int,
    supercell_trafo: Sequence[int],
) -> Any:
    """Create an energy function that also returns neighbor count and violations.

    Used internally by `create_neuralil_kernel_set` for the
    adjust_and_run violation detection flow.

    Returns:
        A callable (positions, types, cell) -> (energy, neighbors, violations_dict).
    """
    _require_neuralil()

    model_info, dynamics_model = load_neuralil_model(
        pickle_file, max_neighbors, supercell_trafo
    )

    s_a, s_b, s_c = supercell_trafo
    r_cut = model_info.r_cut
    model_params = model_info.params

    def get_neighbors(positions, types, cell):
        return _get_max_number_of_neighbors(
            positions, types, r_cut, cell, s_a, s_b, s_c
        )

    def checked_energy(positions, types, cell):
        neighbors = get_neighbors(positions, types, cell)

        energy_ensemble = dynamics_model.apply(
            model_params,
            positions,
            types,
            cell,
            method=dynamics_model.calc_potential_energy,
        )
        energy = energy_ensemble.mean()

        violations = {
            "max_neighbors": neighbors > max_neighbors,
        }

        return energy, neighbors, violations

    return checked_energy


# ---------------------------------------------------------------------------
# Public API: create_neuralil (for loader.py)
# ---------------------------------------------------------------------------


def create_neuralil(
    pickle_file: str | None = None,
    max_neighbors: int = 50,
    supercell_trafo: Sequence[int] = (1, 1, 1),
    **kwargs: Any,
) -> tuple[Any, Params]:
    """Create a NeuralIL energy backend.

    This is the entry point used by `load_backend("neuralil", ...)`.

    Args:
        pickle_file: Path to the NeuralIL model pickle file.
        max_neighbors: Compile-time max neighbor count.
        supercell_trafo: Supercell diagonal transformation.
        **kwargs: Unused, for forward compatibility.

    Returns:
        (energy_fn, params) tuple conforming to EnergyFn protocol.

    Raises:
        ImportError: If NeuralIL is not installed.
        ValueError: If pickle_file is not provided.
    """
    _require_neuralil()

    if pickle_file is None:
        raise ValueError(
            "pickle_file is required for the neuralil backend. "
            "Pass the path to your NeuralIL model pickle."
        )

    return _make_energy_fn(pickle_file, max_neighbors, supercell_trafo)


# ---------------------------------------------------------------------------
# Public API: create_neuralil_kernel_set (for multi-kernel dispatch)
# ---------------------------------------------------------------------------


def create_neuralil_kernel_set(
    pickle_file: str,
    max_neighbors_list: list[int],
    supercell_trafo: Sequence[int] = (1, 1, 1),
    max_neighbors_offset: int = 5,
) -> Any:
    """Create a CompiledKernelSet for NeuralIL with multi-kernel dispatch.

    Pre-compiles one energy function per max_neighbors bucket. Use with
    `kernel_set.adjust_and_run()` for automatic violation detection and
    retry escalation.

    Args:
        pickle_file: Path to NeuralIL model pickle.
        max_neighbors_list: List of max_neighbors values to pre-compile.
        supercell_trafo: Supercell diagonal transformation.
        max_neighbors_offset: Safety margin for bucket selection.

    Returns:
        CompiledKernelSet instance.
    """
    _require_neuralil()

    from jaxrens.backends.kernel_dispatch import CompiledKernelSet

    def kernel_factory(max_neighbors: int) -> Any:
        energy_fn, _ = _make_energy_fn(
            pickle_file, max_neighbors, supercell_trafo
        )
        return energy_fn

    return CompiledKernelSet(
        kernel_factory=kernel_factory,
        max_neighbors_list=max_neighbors_list,
        max_neighbors_offset=max_neighbors_offset,
    )


def is_available() -> bool:
    """Check whether NeuralIL is installed."""
    return _NEURALIL_AVAILABLE
