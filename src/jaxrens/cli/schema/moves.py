"""Pydantic schema for the [moves] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields that its kernel builder
accepts.  ``to_move_config()`` and ``to_descriptor()`` are the seam between
the CLI config layer and the library core — they replace the
``_MOVE_REGISTRY`` / ``_build_kernel_kwargs`` / ``_extra_state_fields``
side-channel that used to live in ``cli/run.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import (
    alchemical,
    galilean,
    hmc,
    random_walk,
    shear,
    single_atom,
    stretch,
    volume,
)
from jaxrens.state.config import MoveConfig

if TYPE_CHECKING:
    from jaxrens.cli.schema.cell import CellSpec

# ---------------------------------------------------------------------------
# MoveType literal — kept for backward compatibility with callers that do
# ``from jaxrens.cli.schema.moves import MoveType``.
# ---------------------------------------------------------------------------

MoveType = Literal[
    "random_walk",
    "gmc",
    "hmc",
    "single_atom",
    "single_atom_sweep",
    "single_atom_swap",
    "volume",
    "shear",
    "stretch",
    "alchemical_morph",
    "alchemical_shift",
]


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------


class BaseMoveSpec(BaseModel):
    """Fields shared by every move type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_size: float = 0.1
    weight: float = 1.0
    adaptation_warmup: int = 100
    target_acceptance: float = 0.5
    name: str | None = None

    @property
    def move_type(self) -> str:
        """Backward-compatible alias for the ``type`` discriminator field."""
        return self.type  # type: ignore[attr-defined]

    def _effective_name(self) -> str:
        return self.name if self.name is not None else self.type  # type: ignore[attr-defined]

    def to_move_config(self) -> MoveConfig:
        """Produce the library ``MoveConfig`` dataclass."""
        return MoveConfig(
            move_type=self.type,  # type: ignore[attr-defined]
            step_size=self.step_size,
            n_steps=self._n_steps(),
            weight=self.weight,
            adaptation_warmup=self.adaptation_warmup,
            target_acceptance=self.target_acceptance,
        )

    def _n_steps(self) -> int:
        """Override in subclasses that carry a steps-like field."""
        return 10

    def _build_kernel(self) -> Callable:
        raise NotImplementedError

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        *,
        reference_positions: Any = None,
        reference_cell: Any = None,
        backend: Any = None,
    ) -> dict[str, Any]:
        """Return kernel keyword arguments.

        Simple move specs ignore every argument.  Cell-move specs (volume,
        shear, stretch) and sweep specs use ``n_atoms``/``cell_cfg`` to
        populate cell-geometry bounds from the resolver-provided values
        rather than duplicating those fields on the spec. Local-update-
        capable specs (``single_atom_swap``, ``alchemical_morph``) use
        ``reference_positions``/``reference_cell``/``backend`` to build a
        static neighbor table (see ``sampling/neighbor_list.py``) when
        ``local_update: true`` is set.

        Args:
            n_atoms: Number of atoms, derived from the resolved initial
                positions.  ``None`` is accepted by specs that don't need it.
            cell_cfg: ``CellSpec`` carrying cell-geometry constraints.
                ``None`` is accepted by specs that don't need it.
            reference_positions: The resolved initial positions for one
                walker (host array), used only by local-update-capable
                specs to build a fixed-geometry neighbor table.
            reference_cell: The resolved initial cell for one walker,
                paired with ``reference_positions``.
            backend: The resolved (possibly ``EnsembleBackend``-wrapped)
                energy backend, used only to read ``r_cutoff``/probe local
                energy capability when building a static neighbor table.
        """
        return {}

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {}

    def _reject_reasons(self) -> frozenset[str]:
        """Return the set of reject-reason buckets this move can emit.

        Subclasses override this when their kernel can emit cell or prior
        rejection reasons (buckets 2 and 3 respectively).  The default is
        energy-only rejection (bucket 1), which covers all atom-displacement
        moves that never call check_cell_shape or sample from a prior.
        """
        return frozenset({"energy"})

    def _mutates(self) -> frozenset[str]:
        """Return the state aspects this move writes (see jaxrens.constraints).

        Drives which configuration constraints gate this move.  The default
        is ``{"positions"}`` — atom-displacement moves.  Cell moves override
        to add ``"cell"`` (they co-transform atoms, so they keep
        ``"positions"`` too), and species-changing moves override to
        ``{"types"}``.
        """
        return frozenset({"positions"})

    def _affects(self) -> str:
        """Structural locality declaration for ``MoveKernel.affects``.

        ``"all"`` by default — override to ``"local"`` only for moves that
        structurally touch a bounded, small set of atoms (see the
        ``MoveKernel.affects`` docstring). Independent of whether
        ``local_update`` is actually enabled; ``to_descriptor`` combines
        the two.
        """
        return "all"

    def to_descriptor(
        self,
        *,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        reference_positions: Any = None,
        reference_cell: Any = None,
        backend: Any = None,
    ) -> MoveKernel:
        """Produce the ``MoveKernel`` for ``build_mwg``.

        Args:
            n_atoms: Number of atoms, derived from the resolved initial
                positions at resolver time.  Simple moves (random_walk,
                galilean, …) ignore this.  Cell-move specs (volume, shear,
                stretch) and single_atom_sweep use it to populate
                ``kernel_kwargs["n_atoms"]``.
            cell_cfg: ``CellSpec`` carrying cell-geometry constraints.
                Cell-move specs use it to populate ``max_volume_per_atom``,
                ``min_volume_per_atom``, ``min_aspect_ratio``, and
                ``flat_V_prior`` in ``kernel_kwargs``.  Simple moves ignore it.
            reference_positions, reference_cell, backend: Forwarded to
                ``_kernel_kwargs`` — see its docstring. Ignored by every
                move spec that isn't local-update-capable.
        """
        return MoveKernel(
            name=self._effective_name(),
            build_kernel=self._build_kernel(),
            kernel_kwargs=self._kernel_kwargs(
                n_atoms=n_atoms,
                cell_cfg=cell_cfg,
                reference_positions=reference_positions,
                reference_cell=reference_cell,
                backend=backend,
            ),
            weight=self.weight,
            step_size=self.step_size,
            extra_state_fields=self._extra_state_fields(),
            reject_reasons=self._reject_reasons(),
            mutates=self._mutates(),
            affects=self._affects(),
        )


def _build_local_update_kwargs(
    *,
    spec_name: str,
    backend: Any,
    reference_positions: Any,
    reference_cell: Any,
    max_neighbors: int | None,
    max_neighbors_margin: int,
) -> dict[str, Any]:
    """Shared ``local_update: true`` wiring for all four local moves.

    Resolves ``max_neighbors`` from the backend's own coordination-number
    probe when not given explicitly (used by each move spec to derive its
    own default ``max_affected``, since the right formula differs by move
    — whether the touched atom's position itself changes).

    The periodic-image geometry itself is NOT built here any more: every
    local move kernel now rebuilds it at every proposal from the CURRENT
    ``state.image_bucket`` (a static field managed by the image-count
    bucket ladder — see ``sampling/bucket_manager.py`` and
    ``sampling/neighbor_list.py::build_symmetric_image_offsets``), so it
    stays correct even if the cell changes after this resolve-time call
    (a volume move, or per-walker cell diversity). This function's only
    remaining geometry-independent job is resolving ``max_neighbors``.
    """
    if backend is None:
        raise ValueError(
            f"{spec_name}: local_update=True requires the resolver to "
            "supply backend to to_descriptor() (internal wiring error if "
            "this is user-visible)."
        )
    if not hasattr(backend, "atomic_energies_for"):
        raise ValueError(
            f"{spec_name}: local_update=True requires a backend that "
            f"implements atomic_energies_for (e.g. NeuralIL); got "
            f"{type(backend).__name__}."
        )

    resolved_max_neighbors = max_neighbors
    if resolved_max_neighbors is None:
        probe = getattr(backend, "max_neighbors_for", None)
        if (
            probe is None
            or reference_positions is None
            or reference_cell is None
        ):
            raise ValueError(
                f"{spec_name}: local_update=True needs max_neighbors set "
                "explicitly, since the backend has no max_neighbors_for "
                "to derive a default from."
            )
        resolved_max_neighbors = int(
            probe(reference_positions, reference_cell)
        ) + (max_neighbors_margin)

    return {"_resolved_max_neighbors": resolved_max_neighbors}


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------


class RandomWalkMoveSpec(BaseMoveSpec):
    type: Literal["random_walk"] = "random_walk"

    def _build_kernel(self) -> Callable:
        return random_walk.build_kernel


class GMCMoveSpec(BaseMoveSpec):
    """Galilean Monte Carlo move.

    The legacy YAML key ``type: galilean`` is accepted via a pre-validator
    coercion in ``root.py::_coerce_move_dict`` and rewritten to ``type: gmc``
    at parse time.
    """

    type: Literal["gmc"] = "gmc"
    n_reflect: int = 5

    def _n_steps(self) -> int:
        return self.n_reflect

    def _build_kernel(self) -> Callable:
        return galilean.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        return {"n_reflect": self.n_reflect}

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {
            "direction": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros_like(positions),
            ),
        }


class HMCMoveSpec(BaseMoveSpec):
    type: Literal["hmc"] = "hmc"
    n_leapfrog: int = 10

    def _n_steps(self) -> int:
        return self.n_leapfrog

    def _build_kernel(self) -> Callable:
        return hmc.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        return {"n_leapfrog": self.n_leapfrog}


class SingleAtomMoveSpec(BaseMoveSpec):
    type: Literal["single_atom"] = "single_atom"
    local_update: bool = False
    max_neighbors: int | None = None
    max_affected: int | None = None

    def _build_kernel(self) -> Callable:
        return single_atom.build_kernel

    def _affects(self) -> str:
        return "local" if self.local_update else "all"

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        if not self.local_update:
            return {}
        return {
            "atomic_energies": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros(positions.shape[0]),
            ),
            "raw_energy": (
                jnp.ndarray,
                lambda positions, types: jnp.asarray(0.0),
            ),
        }

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        *,
        reference_positions: Any = None,
        reference_cell: Any = None,
        backend: Any = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if not self.local_update:
            return {}
        built = _build_local_update_kwargs(
            spec_name="single_atom",
            backend=backend,
            reference_positions=reference_positions,
            reference_cell=reference_cell,
            max_neighbors=self.max_neighbors,
            max_neighbors_margin=4,
        )
        resolved_max_neighbors = built.pop("_resolved_max_neighbors")
        # The moved atom + neighbors under its old position + neighbors
        # under its new position (a union, so up to 2x a typical
        # coordination number in the worst case where the two sets are
        # disjoint).
        default_max_affected = 1 + 2 * resolved_max_neighbors
        max_affected = (
            self.max_affected
            if self.max_affected is not None
            else default_max_affected
        )
        return {**built, "max_affected": max_affected}


class SingleAtomSweepMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_sweep"] = "single_atom_sweep"
    local_update: bool = False
    max_neighbors: int | None = None
    max_affected: int | None = None

    def _build_kernel(self) -> Callable:
        return single_atom.build_sweep_kernel

    def _affects(self) -> str:
        return "local" if self.local_update else "all"

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        if not self.local_update:
            return {}
        return {
            "atomic_energies": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros(positions.shape[0]),
            ),
            "raw_energy": (
                jnp.ndarray,
                lambda positions, types: jnp.asarray(0.0),
            ),
        }

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        *,
        reference_positions: Any = None,
        reference_cell: Any = None,
        backend: Any = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "SingleAtomSweepMoveSpec.to_descriptor() requires n_atoms to "
                "be provided by the resolver (derived from init positions). "
                "Call to_descriptor(n_atoms=...) with the atom count."
            )
        kwargs: dict[str, Any] = {"n_atoms": n_atoms}
        if not self.local_update:
            return kwargs
        built = _build_local_update_kwargs(
            spec_name="single_atom_sweep",
            backend=backend,
            reference_positions=reference_positions,
            reference_cell=reference_cell,
            max_neighbors=self.max_neighbors,
            max_neighbors_margin=4,
        )
        resolved_max_neighbors = built.pop("_resolved_max_neighbors")
        default_max_affected = 1 + 2 * resolved_max_neighbors
        max_affected = (
            self.max_affected
            if self.max_affected is not None
            else default_max_affected
        )
        return {**kwargs, **built, "max_affected": max_affected}


class SingleAtomSwapMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_swap"] = "single_atom_swap"
    local_update: bool = False
    max_neighbors: int | None = None
    max_affected: int | None = None

    def _build_kernel(self) -> Callable:
        return single_atom.build_swap_kernel

    def _mutates(self) -> frozenset[str]:
        return frozenset({"types"})

    def _affects(self) -> str:
        return "local" if self.local_update else "all"

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        if not self.local_update:
            return {}
        return {
            "atomic_energies": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros(positions.shape[0]),
            ),
            "raw_energy": (
                jnp.ndarray,
                lambda positions, types: jnp.asarray(0.0),
            ),
        }

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        *,
        reference_positions: Any = None,
        reference_cell: Any = None,
        backend: Any = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if not self.local_update:
            return {}
        built = _build_local_update_kwargs(
            spec_name="single_atom_swap",
            backend=backend,
            reference_positions=reference_positions,
            reference_cell=reference_cell,
            max_neighbors=self.max_neighbors,
            max_neighbors_margin=4,
        )
        resolved_max_neighbors = built.pop("_resolved_max_neighbors")
        # Two touched atoms, positions unchanged (no old/new doubling).
        default_max_affected = 2 + 2 * resolved_max_neighbors
        max_affected = (
            self.max_affected
            if self.max_affected is not None
            else default_max_affected
        )
        return {**built, "max_affected": max_affected}


class VolumeMoveSpec(BaseMoveSpec):
    type: Literal["volume"] = "volume"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell", "prior"})

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return volume.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "VolumeMoveSpec.to_descriptor() requires n_atoms to be "
                "provided by the resolver (derived from init positions)."
            )
        if cell_cfg is None:
            raise ValueError(
                "VolumeMoveSpec.to_descriptor() requires cell_cfg to be "
                "provided by the resolver (from the [cell] config section)."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
            "flat_v_prior": cell_cfg.flat_V_prior,
        }


class ShearMoveSpec(BaseMoveSpec):
    type: Literal["shear"] = "shear"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell"})

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return shear.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "ShearMoveSpec.to_descriptor() requires n_atoms to be "
                "provided by the resolver (derived from init positions)."
            )
        if cell_cfg is None:
            raise ValueError(
                "ShearMoveSpec.to_descriptor() requires cell_cfg to be "
                "provided by the resolver (from the [cell] config section)."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
        }


class StretchMoveSpec(BaseMoveSpec):
    type: Literal["stretch"] = "stretch"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell"})

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return stretch.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "StretchMoveSpec.to_descriptor() requires n_atoms to be "
                "provided by the resolver (derived from init positions)."
            )
        if cell_cfg is None:
            raise ValueError(
                "StretchMoveSpec.to_descriptor() requires cell_cfg to be "
                "provided by the resolver (from the [cell] config section)."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
        }


class AlchemicalMorphMoveSpec(BaseMoveSpec):
    type: Literal["alchemical_morph"] = "alchemical_morph"
    n_species: int
    local_update: bool = False
    max_neighbors: int | None = None
    max_affected: int | None = None
    # NOTE: n_species could in principle be derived from len(symbol_map) in
    # init_resolved, but that would require threading symbol_map through the
    # resolver to to_descriptor().  Since it is single-valued and small, keeping
    # it on the spec is a pragmatic trade-off; the inconsistency is flagged here.

    def _build_kernel(self) -> Callable:
        return alchemical.build_morph_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        *,
        reference_positions: Any = None,
        reference_cell: Any = None,
        backend: Any = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"n_species": self.n_species}
        if not self.local_update:
            return kwargs
        built = _build_local_update_kwargs(
            spec_name="alchemical_morph",
            backend=backend,
            reference_positions=reference_positions,
            reference_cell=reference_cell,
            max_neighbors=self.max_neighbors,
            max_neighbors_margin=4,
        )
        resolved_max_neighbors = built.pop("_resolved_max_neighbors")
        # One touched atom, position unchanged (no old/new doubling).
        default_max_affected = 1 + resolved_max_neighbors
        max_affected = (
            self.max_affected
            if self.max_affected is not None
            else default_max_affected
        )
        return {**kwargs, **built, "max_affected": max_affected}

    def _mutates(self) -> frozenset[str]:
        return frozenset({"types"})

    def _affects(self) -> str:
        return "local" if self.local_update else "all"

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        if not self.local_update:
            return {}
        return {
            "atomic_energies": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros(positions.shape[0]),
            ),
            "raw_energy": (
                jnp.ndarray,
                lambda positions, types: jnp.asarray(0.0),
            ),
        }


class AlchemicalShiftMoveSpec(BaseMoveSpec):
    type: Literal["alchemical_shift"] = "alchemical_shift"
    assume_translation_invariant: bool = False

    def _build_kernel(self) -> Callable:
        return alchemical.build_shift_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        return {
            "assume_translation_invariant": self.assume_translation_invariant
        }


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

MoveSpec = Annotated[
    Union[
        RandomWalkMoveSpec,
        GMCMoveSpec,
        HMCMoveSpec,
        SingleAtomMoveSpec,
        SingleAtomSweepMoveSpec,
        SingleAtomSwapMoveSpec,
        VolumeMoveSpec,
        ShearMoveSpec,
        StretchMoveSpec,
        AlchemicalMorphMoveSpec,
        AlchemicalShiftMoveSpec,
    ],
    Field(discriminator="type"),
]
