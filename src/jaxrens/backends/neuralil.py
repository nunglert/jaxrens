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

from jaxrens.backends.base import BackendResult
from jaxrens.backends.softcore import DEFAULT_SOFTCORE_KWARGS

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

try:
    from neuralil.softcore.model import SoftCoreNeuralIL, SoftCorePlainEnsemble

    _SOFTCORE_AVAILABLE = True
except ImportError:
    SoftCoreNeuralIL = None  # type: ignore[assignment]
    SoftCorePlainEnsemble = None  # type: ignore[assignment]
    _SOFTCORE_AVAILABLE = False


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
    morse_type: str = "RepulsiveMorse",
    softcore: bool = False,
    softcore_kwargs: dict[str, float] | None = None,
):
    """Build the (max_neighbors-independent) NeuralIL dynamics model.

    Returns a single ``NeuralIL[withMorse]`` when ``is_ensemble`` is
    False, otherwise the corresponding ``PlainEnsemble[withMorse]``
    wrapper. ``morse_type`` selects the Morse flavour (``"RepulsiveMorse"``
    or ``"Morse"``); ignored when ``has_morse`` is False.

    When ``softcore=True`` the dynamics model is a ``SoftCoreNeuralIL``
    (or ``SoftCorePlainEnsemble`` wrapper) that augments the plain
    ``NeuralIL`` with a fixed repulsive Morse term parameterised by
    ``softcore_kwargs`` (``a0``, ``b0``, ``d0``, ``r_core_cut``,
    ``r_core_switch``). Mutually exclusive with the trainable ``has_morse``
    branch.
    """
    if softcore and has_morse:
        raise ValueError(
            "softcore and has_morse are mutually exclusive: the soft-core "
            "model adds its own fixed repulsive Morse term, so the trainable "
            "Morse branch must be disabled."
        )
    if softcore and not _SOFTCORE_AVAILABLE:
        raise ImportError(
            "softcore=True requires neuralil.softcore.model, which was not "
            "importable. Update your neuralil install to a version that "
            "ships the softcore subpackage."
        )

    descriptor_gen = PowerSpectrumGenerator(
        n_max,
        r_cutoff,
        n_types,
        supercell_trafo,
    )
    core_model = ResNetCore(core_widths)

    if softcore:
        sc_kwargs = dict(DEFAULT_SOFTCORE_KWARGS)
        if softcore_kwargs is not None:
            sc_kwargs.update(softcore_kwargs)
        individual = SoftCoreNeuralIL(
            n_types,
            embed_d,
            r_cutoff,
            descriptor_gen,
            descriptor_gen.process_some_data,
            core_model,
            **sc_kwargs,
        )
    elif has_morse:
        individual = NeuralILwithMorse(
            n_types,
            embed_d,
            r_cutoff,
            descriptor_gen,
            descriptor_gen.process_some_data,
            core_model,
            morse_type=morse_type,
        )
    else:
        individual = NeuralIL(
            n_types,
            embed_d,
            r_cutoff,
            descriptor_gen,
            descriptor_gen.process_some_data,
            core_model,
        )

    if not is_ensemble:
        return individual

    if softcore:
        return SoftCorePlainEnsemble(individual, n_ensemble)
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
        morse_type: str = "RepulsiveMorse",
        softcore: bool = False,
        softcore_kwargs: dict[str, float] | None = None,
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
        self.morse_type = morse_type
        self.softcore = softcore
        self.softcore_kwargs = (
            None if softcore_kwargs is None else dict(softcore_kwargs)
        )
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
            morse_type=morse_type,
            softcore=softcore,
            softcore_kwargs=self.softcore_kwargs,
        )

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        """Atomic numbers (Z) of the elements the model was trained on,
        in the same order as ``sorted_elements``. Used by the resolver to
        map user-supplied Z numbers in ``start_species`` to the model's
        contiguous 0-based type indices."""
        from ase.data import atomic_numbers as _Z

        return tuple(_Z[s] for s in self.sorted_elements)

    @staticmethod
    def _safe_cell(cell: jnp.ndarray) -> jnp.ndarray:
        """Substitute a large dummy cell for non-periodic (zero-trace) inputs."""
        return jnp.where(jnp.trace(cell) == 0, 1000.0 * jnp.eye(3), cell)

    def _apply_energy_shift(
        self, energy: jnp.ndarray, species: jnp.ndarray
    ) -> jnp.ndarray:
        """Re-add the per-atom baseline that training subtracted from targets.
        This will become important for grand-canonical formulations.

        A constant per-atom shift; it does not affect forces.
        """
        if self.energy_shift_per_atom == 0.0:
            return energy
        n_real = (species >= 0).sum()
        return energy + self.energy_shift_per_atom * n_real

    def _neighbor_diagnostics(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        safe_cell: jnp.ndarray,
        max_neighbors: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Actual max neighbor count + overflow flag for the bucket manager."""
        sc_a, sc_b, sc_c = self.supercell_trafo
        actual_max_neighbors = _get_max_number_of_neighbors(
            positions,
            species,
            self.r_cutoff,
            safe_cell,
            sc_a,
            sc_b,
            sc_c,
        )
        overflow = actual_max_neighbors > max_neighbors
        return actual_max_neighbors, overflow

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        safe_cell = self._safe_cell(cell)

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
        energy = self._apply_energy_shift(energy, species)

        actual_max_neighbors, overflow = self._neighbor_diagnostics(
            positions, species, safe_cell, max_neighbors
        )

        return BackendResult(
            energy=energy,
            max_neighbor_count=actual_max_neighbors,
            overflow=overflow,
        )

    def energy_and_forces(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Native energy + forces via NeuralIL's ``calc_potential_energy_and_forces``.

        Every dynamics-model variant (single / Morse / soft-core / ensemble)
        exposes ``calc_potential_energy_and_forces``, returning ``(energy,
        forces)`` with ``forces = -dE/dx``. The soft-core and Morse terms are
        included because the differentiated ``calc_potential_energy`` runs
        through the same (overridden) ``calc_atomic_energies`` as ``__call__``.

        For an ensemble the model returns per-member energies ``(n_ensemble,)``
        and forces ``(n_ensemble, N, 3)``; both are reduced by mean to match the
        committee prediction used in ``__call__``. The per-atom energy shift is a
        constant and is added to the energy only — it does not affect forces.
        """
        safe_cell = self._safe_cell(cell)

        energy_out, forces_out = self._dynamics_model.apply(
            self.model_params,
            positions,
            species,
            safe_cell,
            max_neighbors,
            method=self._dynamics_model.calc_potential_energy_and_forces,
        )
        if self.is_ensemble:
            energy = energy_out.mean()
            forces = forces_out.mean(axis=0)
        else:
            energy = energy_out
            forces = forces_out
        energy = self._apply_energy_shift(energy, species)

        actual_max_neighbors, overflow = self._neighbor_diagnostics(
            positions, species, safe_cell, max_neighbors
        )

        return BackendResult(
            energy=energy,
            forces=forces,
            max_neighbor_count=actual_max_neighbors,
            overflow=overflow,
        )

    def members(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Per-committee-member energies and forces (for uncertainty).

        Like :meth:`energy_and_forces` but **keeps the per-ensemble-member
        axis**: populates the reserved ``energy_members`` ``(M,)`` and
        ``forces_members`` ``(M, N, 3)`` slots, in addition to the reduced
        ``energy`` / ``forces`` committee means and the control fields.

        For a single (non-ensemble) model ``M = 1`` (a leading axis is added)
        so the ``(M, …)`` shape contract holds uniformly and the committee
        spread is exactly zero. The per-atom energy shift is added to every
        member, preserving the invariant ``energy == energy_members.mean()``
        (the shift is constant across members, so it does not affect the std).
        """
        safe_cell = self._safe_cell(cell)

        energy_out, forces_out = self._dynamics_model.apply(
            self.model_params,
            positions,
            species,
            safe_cell,
            max_neighbors,
            method=self._dynamics_model.calc_potential_energy_and_forces,
        )
        if self.is_ensemble:
            energy_members = energy_out  # (M,)
            forces_members = forces_out  # (M, N, 3)
        else:
            energy_members = energy_out[jnp.newaxis]  # (1,)
            forces_members = forces_out[jnp.newaxis]  # (1, N, 3)

        # Shift each member equally (constant → no effect on std), keeping
        # energy == energy_members.mean().
        energy_members = self._apply_energy_shift(energy_members, species)
        energy = energy_members.mean()
        forces = forces_members.mean(axis=0)

        actual_max_neighbors, overflow = self._neighbor_diagnostics(
            positions, species, safe_cell, max_neighbors
        )

        return BackendResult(
            energy=energy,
            forces=forces,
            max_neighbor_count=actual_max_neighbors,
            overflow=overflow,
            energy_members=energy_members,
            forces_members=forces_members,
        )

    def energy_members(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        """Per-committee-member total energies, shape ``(M,)`` — no forces.

        The cheap path for energy-only uncertainty: skips the force jacobian
        that :meth:`members` computes. Single (non-ensemble) model → ``(1,)``.
        The per-atom shift is added to every member (constant; irrelevant to
        the committee std, kept for absolute-scale consistency).
        """
        safe_cell = self._safe_cell(cell)
        energy_out = self._dynamics_model.apply(
            self.model_params,
            positions,
            species,
            safe_cell,
            max_neighbors,
            method=self._dynamics_model.calc_potential_energy,
        )
        members = energy_out if self.is_ensemble else energy_out[jnp.newaxis]
        return self._apply_energy_shift(members, species)

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
            jnp.trace(cell) == 0,
            1000.0 * jnp.eye(3),
            cell,
        )
        species = jnp.zeros(positions.shape[0], dtype=jnp.int32)
        return _get_max_number_of_neighbors(
            positions,
            species,
            self.r_cutoff,
            safe_cell,
            sc_a,
            sc_b,
            sc_c,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_neuralil(
    pickle_file: str | None = None,
    supercell_trafo: Sequence[int] = (1, 1, 1),
    softcore: bool | None = None,
    softcore_kwargs: dict[str, float] | None = None,
    **kwargs: Any,
) -> NeuralILBackend:
    """Create a NeuralIL energy backend.

    Auto-detects whether the pickle is a single ``NeuralIL[withMorse]``
    or a ``PlainEnsemble[withMorse]`` from the saved params layout, and
    builds the matching dynamics model.

    The soft-core augmentation (a fixed repulsive Morse term that
    smoothly switches off between ``r_core_switch`` and ``r_core_cut``)
    is *not* auto-detectable from the saved params since it carries no
    trainable parameters. It is opt-in via the pickle's
    ``constructor_kwargs['softcore']`` or via an explicit
    ``softcore=True`` override here; ``softcore_kwargs`` overrides the
    pickle-stored kwargs (merged on top of the package defaults).

    Args:
        pickle_file: Path to NeuralIL model pickle (``NeuralILModelInfo``).
            Either single-model or ensemble checkpoints are accepted.
        supercell_trafo: Supercell diagonal transformation (s_a, s_b, s_c).
        softcore: If True, wrap the model with a fixed repulsive Morse
            soft-core term. ``None`` (default) means "use whatever the
            pickle says"; explicit ``True``/``False`` overrides the pickle.
            Mutually exclusive with a trainable Morse model.
        softcore_kwargs: Optional override of the soft-core hyperparameters
            (``a0``, ``b0``, ``d0``, ``r_core_cut``, ``r_core_switch``).
            Merged on top of the package defaults
            (``DEFAULT_SOFTCORE_KWARGS``) and on top of any kwargs read
            from the pickle.

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

    # The Morse flavour (full ``MorseModel`` vs purely-repulsive
    # ``RepulsiveMorseModel``) is *not* recoverable from the flax params
    # — both classes share the same parameter tree.  Training writes it
    # into ``constructor_kwargs``; we read it back here and pass it to
    # ``NeuralILwithMorse(...)``.  Older pickles have
    # ``constructor_kwargs={}``: fall back to the new default and warn.
    ckwargs = getattr(model_info, "constructor_kwargs", None) or {}
    if has_morse and "morse_type" not in ckwargs:
        logger.warning(
            "NeuralIL pickle has no 'morse_type' in constructor_kwargs; "
            "defaulting to 'RepulsiveMorse'. If this model was trained "
            "with the full MorseModel, pass morse_type='Morse' or "
            "edit the pickle's constructor_kwargs."
        )
    morse_type = ckwargs.get("morse_type", "RepulsiveMorse")

    # Soft-core: pickle is the source of truth unless the caller overrides.
    pickle_softcore = bool(ckwargs.get("softcore", False))
    use_softcore = pickle_softcore if softcore is None else bool(softcore)
    merged_softcore_kwargs: dict[str, float] = {}
    pickle_softcore_kwargs = ckwargs.get("softcore_kwargs")
    if isinstance(pickle_softcore_kwargs, dict):
        merged_softcore_kwargs.update(pickle_softcore_kwargs)
    if softcore_kwargs is not None:
        merged_softcore_kwargs.update(softcore_kwargs)
    final_softcore_kwargs = merged_softcore_kwargs or None

    logger.info(
        "NeuralIL backend created: r_cut=%.2f, elements=%s, supercell=%s, "
        "ensemble=%s, n_ensemble=%d, morse=%s, morse_type=%s, "
        "softcore=%s, softcore_kwargs=%s, energy_shift_per_atom=%g",
        model_info.r_cut,
        model_info.sorted_elements,
        supercell_trafo,
        is_ensemble,
        n_ensemble,
        has_morse,
        morse_type,
        use_softcore,
        final_softcore_kwargs,
        energy_shift_per_atom,
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
        morse_type=morse_type,
        softcore=use_softcore,
        softcore_kwargs=final_softcore_kwargs,
        energy_shift_per_atom=energy_shift_per_atom,
    )
