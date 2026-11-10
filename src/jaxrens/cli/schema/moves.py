"""Pydantic schema for the [moves] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields that its kernel builder
accepts.  ``to_move_config()`` and ``to_descriptor()`` are the seam between
the CLI config layer and the library core — they replace the
``_MOVE_REGISTRY`` / ``_build_kernel_kwargs`` / ``_extra_state_fields``
side-channel that used to live in ``cli/run.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

import jax.numpy as jnp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import (
    alchemical,
    galilean,
    hmc,
    layered_moves,
    random_walk,
    shear,
    single_atom,
    stretch,
    swap,
    toy_1d,
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
    "species_swap",
    "volume",
    "shear",
    "stretch",
    "alchemical_morph",
]


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------


class BaseMoveSpec(BaseModel):
    """Fields shared by every move type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_size: float = Field(
        default=0.1,
        description=(
            "Initial proposal magnitude, in the move's own units "
            "(Angstrom for displacement moves, a relative fraction for "
            "cell moves).  A starting point only when adaptation is on — "
            "``adaptation.adjust_interval`` then drives it toward "
            "``target_acceptance``."
        ),
    )
    weight: float = Field(
        default=1.0,
        description=(
            "Relative dispatch probability within the "
            "Metropolis-within-Gibbs scheduler.  Weights are normalised "
            "across the whole ``moves:`` list, so only their ratios "
            "matter."
        ),
    )
    adaptation_warmup: int = Field(
        default=100,
        description=(
            "Iterations before this move's step size starts adapting, "
            "letting the acceptance statistics settle first."
        ),
    )
    target_acceptance: float = Field(
        default=0.5,
        description=(
            "Acceptance rate the adaptation aims for.  Ignored by moves "
            "with no continuous magnitude, such as ``species_swap``."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Label for this move.  Defaults to its ``type``.  Keys the "
            "monitor's per-move columns, the adaptation diagnostics, and "
            "``adaptation.per_move`` overrides — so give two moves of the "
            "same type distinct names, otherwise they collide in all "
            "three."
        ),
    )

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
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Return kernel keyword arguments.

        Simple move specs ignore ``n_atoms``, ``cell_cfg`` and ``symbol_map``.
        Cell-move specs (volume, shear, stretch) and sweep specs use the first
        two to populate ``n_atoms`` / cell-geometry bounds from the
        resolver-provided values rather than duplicating those fields on the
        spec; species-scoped specs use the third to map element symbols to
        type codes.

        Args:
            n_atoms: Number of atoms, derived from the resolved initial
                positions.  ``None`` is accepted by specs that don't need it.
            cell_cfg: ``CellSpec`` carrying cell-geometry constraints.
                ``None`` is accepted by specs that don't need it.
            symbol_map: ``{type_code: element_symbol}`` for the resolved
                system.  ``None`` is accepted by specs that don't need it.
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

    def to_descriptor(
        self,
        *,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
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
            symbol_map: ``{type_code: element_symbol}`` for the resolved
                system, as carried by ``ResolvedInit``.  Only species-scoped
                specs (``gmc`` with ``species``) need it; all others ignore it.
        """
        return MoveKernel(
            name=self._effective_name(),
            build_kernel=self._build_kernel(),
            kernel_kwargs=self._kernel_kwargs(
                n_atoms=n_atoms, cell_cfg=cell_cfg, symbol_map=symbol_map
            ),
            weight=self.weight,
            step_size=self.step_size,
            extra_state_fields=self._extra_state_fields(),
            reject_reasons=self._reject_reasons(),
            mutates=self._mutates(),
        )


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------


def _wrap_bare_symbol(v: Any) -> Any:
    """Accept ``species: Li`` as shorthand for ``species: [Li]``."""
    return (v,) if isinstance(v, str) else v


def _restrict_kwargs(cell_cfg: "CellSpec") -> dict[str, Any]:
    """``restrict_c_in_plane`` for the cell-move kernels, only when set.

    Left out when false, so cell-move descriptors stay identical to those
    built before the option existed.
    """
    if cell_cfg.restrict_c_in_plane:
        return {"restrict_c_in_plane": True}
    return {}


def _species_codes(
    species: tuple[str, ...] | None,
    symbol_map: dict[int, str] | None,
    move_label: str,
    spec_label: str,
) -> tuple[int, ...] | None:
    """Resolve element symbols to the contiguous type codes used by types.

    Shared by the species-scoped atom moves.  Returns ``None`` for an
    unscoped move.
    """
    if not species:
        return None
    if symbol_map is None:
        raise ValueError(
            f"{spec_label} with species=... requires symbol_map to be "
            "provided by the resolver (it maps element symbols to the "
            "type codes stored in WalkerState.types). Build descriptors "
            "via ResolvedConfig.move_descriptors rather than "
            "setup_mwg()/MoveConfig, which carry no species information."
        )
    code_of = {sym: code for code, sym in symbol_map.items()}
    unknown = [s for s in species if s not in code_of]
    if unknown:
        raise ValueError(
            f"{move_label} species {unknown} not present in the system "
            f"(symbols: {sorted(code_of)}). A species-scoped move whose "
            f"element is absent would never move anything."
        )
    return tuple(code_of[s] for s in species)


class RandomWalkMoveSpec(BaseMoveSpec):
    """Gaussian displacement of every atom at once.

    The cheapest move available — one energy evaluation per proposal, no
    forces — but it decorrelates slowly, since every atom moves at once and
    a single bad contact rejects the whole configuration.  Useful as a
    low-weight background move alongside ``gmc``.

    ::

        moves:
          - {type: random_walk, step_size: 0.1, weight: 1}
    """

    type: Literal["random_walk"] = Field(
        default="random_walk",
        description="Discriminator selecting this move.",
    )

    def _build_kernel(self) -> Callable:
        return random_walk.build_kernel


class GMCMoveSpec(BaseMoveSpec):
    """Galilean Monte Carlo move, optionally scoped to one element sublattice.

    The legacy YAML key ``type: galilean`` is accepted via a pre-validator
    coercion in ``root.py::_coerce_move_dict`` and rewritten to ``type: gmc``
    at parse time.

    ``species`` restricts the move to the atoms of the named element(s) and
    holds the rest fixed.  Declaring one scoped move per element gives each
    sublattice an *independently adapted* step size, because the MWG sampler
    stores step sizes per move and the adaptation manager bisects each move
    separately.  That matters for systems where one sublattice melts well
    before the other: in a single joint move the step size is capped by
    whichever sublattice is stiffer.

    ::

        moves:
          - {type: gmc, species: Ge, step_size: 0.3, weight: 3}
          - {type: gmc, species: Si, step_size: 0.05, weight: 1}

    Two caveats worth knowing:

    * Step sizes are **not comparable across scopes**.  The direction is a
      unit vector over the moving subspace, so per-atom displacement scales
      as ``step_size / sqrt(3 * n_moving)`` — a minority sublattice takes
      larger per-atom steps at equal nominal step size.  Harmless (each is
      adapted on its own acceptance), but don't read the two numbers as
      being on the same scale.
    * Each scoped move costs a full energy+force call per reflection, so
      two scoped moves are 2x the evaluations of one joint move for the same
      ``n_reflect``.  Use ``weight`` to spend the budget where it pays.
    """

    type: Literal["gmc"] = Field(
        default="gmc", description="Discriminator selecting this move."
    )
    n_reflect: int = Field(
        default=5,
        description=(
            "Reflections per Galilean trajectory.  Each costs one "
            "energy-and-force evaluation, so this multiplies the move's "
            "price; more reflections travel further per proposal."
        ),
    )
    species: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Restrict the move to the named element(s), holding the rest "
            "fixed.  A bare symbol (``species: Ge``) is accepted as "
            "shorthand for a one-element list.  Declaring one scoped move "
            "per element gives each sublattice an independently adapted "
            "step size — worth it when one sublattice melts well before "
            "the other.  Scoped moves are auto-named ``gmc_<symbols>``."
        ),
    )
    refresh_direction: bool = Field(
        default=False,
        description=(
            "Draw a fresh random direction on every GMC call instead of "
            "perturbing the previous one, so the direction only persists "
            "within one trajectory of ``n_reflect`` steps.  Valid either "
            "way; mainly a decorrelation / diagnostic switch."
        ),
    )

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: Ge`` as shorthand for ``species: [Ge]``."""
        return _wrap_bare_symbol(v)

    def _n_steps(self) -> int:
        return self.n_reflect

    def _build_kernel(self) -> Callable:
        return galilean.build_kernel

    def _effective_name(self) -> str:
        """Auto-name scoped moves so they stay distinguishable downstream.

        Move names key the monitor's per-move columns, the adaptation
        diagnostics, and ``adaptation.resolve_for`` overrides — two moves both
        called ``"gmc"`` would collide in all three.
        """
        if self.name is not None:
            return self.name
        if self.species:
            return "gmc_" + "_".join(self.species)
        return self.type

    def _direction_field(self) -> str:
        """MCState field name holding this move's persistent direction.

        Unscoped moves keep the historical ``"direction"`` name (they all act
        on the same full subspace, so sharing is benign, and restarts and
        hand-built ``MoveKernel``s keep working).  Scoped moves each get their
        own field — ``build_mwg`` unions ``extra_state_fields`` by name, so a
        shared field would let the Ge move zero out the Si move's persistent
        direction on every call.
        """
        if not self.species:
            return "direction"
        return f"direction_{self._effective_name()}"

    def _species_codes(
        self, symbol_map: dict[int, str] | None
    ) -> tuple[int, ...] | None:
        """Resolve element symbols to the contiguous type codes used by types."""
        return _species_codes(self.species, symbol_map, "gmc", "GMCMoveSpec")

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"n_reflect": self.n_reflect}
        # Only emitted when set, for the same comparability reason as below.
        if self.refresh_direction:
            kwargs["refresh_direction"] = True
        # Only emit the scoping kwargs when the move is actually scoped: the
        # kernel defaults are exactly ``species=None`` /
        # ``direction_field="direction"``, so an unscoped spec keeps producing
        # the historical single-key kernel_kwargs and stays comparable with
        # descriptors built before species scoping existed.
        codes = self._species_codes(symbol_map)
        if codes is not None:
            kwargs["species"] = codes
            kwargs["direction_field"] = self._direction_field()
        return kwargs

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {
            self._direction_field(): (
                jnp.ndarray,
                lambda positions, types, cell: jnp.zeros_like(positions),
            ),
        }


class HMCMoveSpec(BaseMoveSpec):
    """Hamiltonian Monte Carlo with a leapfrog integrator.

    Draws fresh momenta, integrates the equations of motion for
    ``n_leapfrog`` steps, and accepts on the energy error.  Travels much
    further per proposal than a random walk, at ``n_leapfrog``
    energy-and-force evaluations each.

    ::

        moves:
          - {type: hmc, n_leapfrog: 20, step_size: 0.05, weight: 1}
    """

    type: Literal["hmc"] = Field(
        default="hmc", description="Discriminator selecting this move."
    )
    n_leapfrog: int = Field(
        default=10,
        description=(
            "Leapfrog steps per trajectory.  Each costs one "
            "energy-and-force evaluation; longer trajectories decorrelate "
            "further per proposal but cost proportionally more."
        ),
    )

    def _n_steps(self) -> int:
        return self.n_leapfrog

    def _build_kernel(self) -> Callable:
        return hmc.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        return {"n_leapfrog": self.n_leapfrog}


class SingleAtomMoveSpec(BaseMoveSpec):
    """Displace one randomly chosen atom per proposal.

    Acceptance does not collapse as the system grows — only one atom risks
    a bad contact — so this stays usable at densities where whole-system
    moves are rejected almost always.  It also costs a full energy
    evaluation per atom moved, which is why it is usually a supporting move
    rather than the main one.

    ``species`` restricts the pick to the atoms of the named element(s).
    One scoped move per element gives each its own adapted step size, so a
    mobile species is not held back by a stiff one.  Scoped moves are
    auto-named ``single_atom_<symbols>``, which is also their key in
    ``adaptation.per_move``.

    ::

        moves:
          - {type: single_atom, step_size: 0.2, weight: 1}
          - {type: single_atom, species: Li, step_size: 0.3, weight: 2}
    """

    type: Literal["single_atom"] = Field(
        default="single_atom",
        description="Discriminator selecting this move.",
    )
    species: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Pick the displaced atom only among the named element(s).  A "
            "bare symbol (``species: Li``) is accepted as shorthand for a "
            "one-element list.  A walker with none of these atoms rejects "
            "the proposal.  Scoped moves are auto-named "
            "``single_atom_<symbols>``."
        ),
    )

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: Li`` as shorthand for ``species: [Li]``."""
        return _wrap_bare_symbol(v)

    def _build_kernel(self) -> Callable:
        return single_atom.build_kernel

    def _effective_name(self) -> str:
        """Auto-name scoped moves, as ``GMCMoveSpec`` does."""
        if self.name is not None:
            return self.name
        if self.species:
            return "single_atom_" + "_".join(self.species)
        return self.type

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        # Unscoped specs keep the historical empty kwargs, so their
        # descriptors stay comparable with ones built before scoping.
        codes = _species_codes(
            self.species, symbol_map, "single_atom", "SingleAtomMoveSpec"
        )
        return {} if codes is None else {"species": codes}


class SingleAtomSweepMoveSpec(BaseMoveSpec):
    """Displace every atom once per proposal, in a random order.

    One sweep is ``n_atoms`` single-atom proposals, each accepted or
    rejected on its own — so it gets the good acceptance of
    ``single_atom`` while advancing the whole configuration per dispatch.
    Correspondingly it costs ``n_atoms`` energy evaluations.

    ``n_atoms`` is supplied by the resolver from the initial positions, so
    it is not a YAML key.

    ::

        moves:
          - {type: single_atom_sweep, step_size: 0.2, weight: 1}
    """

    type: Literal["single_atom_sweep"] = Field(
        default="single_atom_sweep",
        description="Discriminator selecting this move.",
    )

    def _build_kernel(self) -> Callable:
        return single_atom.build_sweep_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "SingleAtomSweepMoveSpec.to_descriptor() requires n_atoms to "
                "be provided by the resolver (derived from init positions). "
                "Call to_descriptor(n_atoms=...) with the atom count."
            )
        return {"n_atoms": n_atoms}


class SpeciesSwapMoveSpec(BaseMoveSpec):
    """Exchange the identities of two atoms of *different* species.

    The kernel draws an unlike pair by construction, so no evaluation is
    wasted on a same-species draw and the acceptance rate is not capped by
    the composition.

    ``species`` restricts the exchange to the named elements::

        moves:
          - {type: species_swap, weight: 2}                 # any unlike pair
          - {type: species_swap, species: [Ge, Si]}         # Ge <-> Si only

    A scoped move needs at least two distinct elements — a one-element scope
    has nothing to exchange and is rejected at parse time rather than
    becoming a silent no-op.

    The kernel ignores ``step_size`` (a swap has no continuous magnitude), so
    the inherited ``step_size`` / ``target_acceptance`` fields do nothing
    here.
    """

    type: Literal["species_swap"] = Field(
        default="species_swap",
        description="Discriminator selecting this move.",
    )
    species: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Restrict the exchange to the named elements, e.g. "
            "``[Ge, Si]``.  Must name at least two distinct elements — a "
            "one-element scope has nothing to exchange and is rejected at "
            "parse time.  Omit to allow any unlike pair.  Scoped moves are "
            "auto-named ``species_swap_<symbols>``."
        ),
    )

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: [Ge, Si]``; reject the bare-symbol shorthand.

        ``species: Ge`` is meaningful for ``gmc`` (scope the move to one
        sublattice) but not for a swap, which needs two species to exchange.
        Rejecting it here — rather than letting ``swap.build_kernel`` raise
        much later — puts the error on the YAML key that caused it.
        """
        if isinstance(v, str):
            raise ValueError(
                f"species_swap species must name at least two elements to "
                f"exchange, got the single element {v!r}. Use "
                f"'species: [{v}, <other>]', or drop the key to allow every "
                f"unlike pair."
            )
        return v

    @field_validator("species")
    @classmethod
    def _require_two_distinct(cls, v: Any) -> Any:
        """A swap scope needs two distinct elements; one is a silent no-op."""
        if v is not None and len(set(v)) < 2:
            raise ValueError(
                f"species_swap species={tuple(v)} names {len(set(v))} "
                f"distinct element(s); at least two are needed for a swap."
            )
        return v

    def _build_kernel(self) -> Callable:
        return swap.build_kernel

    def _mutates(self) -> frozenset[str]:
        return frozenset({"types"})

    def _effective_name(self) -> str:
        """Auto-name scoped moves so they stay distinguishable downstream.

        Same rationale as ``GMCMoveSpec._effective_name``: move names key the
        monitor columns, the adaptation diagnostics, and
        ``adaptation.resolve_for`` overrides.
        """
        if self.name is not None:
            return self.name
        if self.species:
            return "species_swap_" + "_".join(self.species)
        return self.type

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        if symbol_map is None:
            raise ValueError(
                "SpeciesSwapMoveSpec requires symbol_map to be provided by "
                "the resolver (it fixes both the number of species and the "
                "symbol -> type-code mapping). Build descriptors via "
                "ResolvedConfig.move_descriptors rather than "
                "setup_mwg()/MoveConfig, which carry no species information."
            )
        kwargs: dict[str, Any] = {"n_species": len(symbol_map)}
        if self.species:
            code_of = {sym: code for code, sym in symbol_map.items()}
            unknown = [s for s in self.species if s not in code_of]
            if unknown:
                raise ValueError(
                    f"species_swap species {unknown} not present in the "
                    f"system (symbols: {sorted(code_of)}). A scoped swap "
                    f"whose element is absent would never accept."
                )
            kwargs["species"] = tuple(code_of[s] for s in self.species)
        return kwargs


class VolumeMoveSpec(BaseMoveSpec):
    """Isotropic cell-volume change, co-transforming atom positions.

    Required for any NPT run — without it the cell never breathes and the
    ``P*V`` term does no work.  ``step_size`` is a relative volume scale,
    not a length, so it is not comparable with the displacement moves.

    Bounds come from the ``cell:`` section, not from this spec.

    ::

        moves:
          - {type: volume, step_size: 0.3, weight: 1}

        cell:
          max_volume_per_atom: 20.0
          min_volume_per_atom: 0.5
    """

    type: Literal["volume"] = Field(
        default="volume", description="Discriminator selecting this move."
    )

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
        symbol_map: dict[int, str] | None = None,
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
            **_restrict_kwargs(cell_cfg),
        }


class ShearMoveSpec(BaseMoveSpec):
    """Volume-preserving shear of the cell, co-transforming positions.

    Lets the cell change shape at fixed volume, which is what allows a
    walker to find a non-cubic crystal structure.  Pair it with
    ``stretch``; neither alone spans the full space of cell shapes.
    Proposals violating ``cell.min_aspect_ratio`` are rejected.

    Bounds come from the ``cell:`` section, not from this spec.

    ::

        moves:
          - {type: shear, step_size: 0.1, weight: 1}
          - {type: stretch, step_size: 0.1, weight: 1}
    """

    type: Literal["shear"] = Field(
        default="shear", description="Discriminator selecting this move."
    )

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
        symbol_map: dict[int, str] | None = None,
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
            **_restrict_kwargs(cell_cfg),
        }


class StretchMoveSpec(BaseMoveSpec):
    """Volume-preserving anisotropic stretch, co-transforming positions.

    Lengthens one cell axis and compresses the others to keep the volume
    fixed.  The complement to ``shear``: together they explore cell shape,
    while ``volume`` explores cell size.

    Bounds come from the ``cell:`` section, not from this spec.

    ::

        moves:
          - {type: stretch, step_size: 0.1, weight: 1}
    """

    type: Literal["stretch"] = Field(
        default="stretch", description="Discriminator selecting this move."
    )

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
        symbol_map: dict[int, str] | None = None,
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
            **_restrict_kwargs(cell_cfg),
        }


class AlchemicalMorphMoveSpec(BaseMoveSpec):
    """Continuously morph one atom's identity toward another species.

    Unlike ``species_swap``, which exchanges two existing atoms and so
    conserves the composition, a morph changes it — use it for semi-grand
    runs where the composition is meant to fluctuate.

    ::

        moves:
          - {type: alchemical_morph, n_species: 2, step_size: 0.1, weight: 1}
    """

    type: Literal["alchemical_morph"] = Field(
        default="alchemical_morph",
        description="Discriminator selecting this move.",
    )
    n_species: int = Field(
        description=(
            "Number of species the morph can select among.  Must match "
            "the species count of the initialised system."
        ),
    )
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
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        return {"n_species": self.n_species}

    def _mutates(self) -> frozenset[str]:
        return frozenset({"types"})


class Lattice1DMoveSpec(BaseMoveSpec):
    """Change the box length of the 1-D toy model.

    The 1-D analogue of a volume move, and the counterpart to
    ``distance_1d``; the RENS paper runs the two in a 1:1 ratio.  Bounds come
    from the ``cell:`` section, read as bounds on ``a / n_atoms``.  Only for
    ``backend: rens_toy``.

    ::

        moves:
          - {type: lattice_1d, step_size: 0.3, weight: 1.0}
          - {type: distance_1d, step_size: 0.3, weight: 1.0}
    """

    type: Literal["lattice_1d"] = Field(
        default="lattice_1d",
        description="Discriminator selecting this move.",
    )

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell"})

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return toy_1d.build_lattice_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        if n_atoms is None or cell_cfg is None:
            raise ValueError(
                "Lattice1DMoveSpec.to_descriptor() requires n_atoms and "
                "cell_cfg from the resolver."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
        }


class Distance1DMoveSpec(BaseMoveSpec):
    """Displace one particle of the 1-D toy model along the axis.

    Counterpart to ``lattice_1d``.  Only ``x`` moves — ``y`` and ``z`` are
    not arguments of the toy energy.  Only for ``backend: rens_toy``.

    ::

        moves:
          - {type: distance_1d, step_size: 0.3, weight: 1.0}
    """

    type: Literal["distance_1d"] = Field(
        default="distance_1d",
        description="Discriminator selecting this move.",
    )

    def _build_kernel(self) -> Callable:
        return toy_1d.build_distance_kernel


# ---------------------------------------------------------------------------
# Layered moves
# ---------------------------------------------------------------------------


class LayeredMoveSpec(BaseMoveSpec):
    """Shared wiring of the moves acting on sheets and galleries.

    The host atoms must form flat sheets parallel to the (a, b) plane of the
    cell.  The resolver checks this on the starting walkers
    (``layered_moves.check_sheets``) and passes the resulting layout to
    ``to_descriptor``.  Every layered move adds the same ``sheet_id`` state
    field, so all of them must use the same ``host_species``.
    """

    host_species: tuple[str, ...] = Field(
        default=("C",),
        description=(
            "Element(s) whose atoms form the sheets, e.g. ``C`` or "
            "``[C, B]`` for doped sheets.  A bare symbol is accepted.  Must "
            "be the same for all layered moves of a run."
        ),
    )
    in_plane_repeats: tuple[int, int] = Field(
        default=(3, 3),
        description=(
            "Number of host primitive cells along a and b.  ``a / m_a`` and "
            "``b / m_b`` are the in-plane lattice vectors of the host.  "
            "Checked against the sheet size, two host atoms per primitive "
            "cell (graphene)."
        ),
    )

    @field_validator("host_species", mode="before")
    @classmethod
    def _wrap_bare_host(cls, v: Any) -> Any:
        """Accept ``host_species: C`` as shorthand for ``[C]``."""
        return _wrap_bare_symbol(v)

    def to_descriptor(
        self,
        *,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
        sheets: "layered_moves.SheetInfo | None" = None,
    ) -> MoveKernel:
        """Produce the ``MoveKernel``, adding the sheet layout.

        Args:
            n_atoms, cell_cfg, symbol_map: As for ``BaseMoveSpec``.
            sheets: Sheet layout of the starting walkers, from
                ``check_sheets``.  Required.
        """
        if sheets is None:
            raise ValueError(
                f"{self.type} needs the sheet layout of the starting walkers, "
                "computed by the resolver. Build descriptors via "
                "ResolvedConfig.move_descriptors."
            )
        m_a, m_b = self.in_plane_repeats
        if 2 * m_a * m_b != sheets.sheet_size:
            raise ValueError(
                f"{self.type}: in_plane_repeats {self.in_plane_repeats} "
                f"implies {2 * m_a * m_b} host atoms per sheet (2 per "
                f"primitive cell), but the sheets have {sheets.sheet_size}."
            )
        desc = super().to_descriptor(
            n_atoms=n_atoms, cell_cfg=cell_cfg, symbol_map=symbol_map
        )
        host_codes = _species_codes(
            self.host_species,
            symbol_map,
            f"{self.type} host",
            type(self).__name__,
        )
        return dataclasses.replace(
            desc,
            kernel_kwargs={
                **desc.kernel_kwargs,
                "host_codes": host_codes,
                "n_sheets": sheets.n_sheets,
                "in_plane_repeats": self.in_plane_repeats,
            },
            extra_state_fields={
                "sheet_id": (
                    jnp.ndarray,
                    layered_moves.make_sheet_id_initializer(host_codes),
                )
            },
        )


class GalleryHopMoveSpec(LayeredMoveSpec):
    """Move one atom to another gallery or another in-plane site.

    Picks one atom of ``species`` and moves it to the equivalent position in
    another gallery (keeping its offset from the gallery centre) and/or by a
    host lattice vector in the plane, plus Gaussian noise of ``step_size``.
    Lets intercalants change gallery without crossing a sheet, which local
    moves cannot do.

    ::

        moves:
          - {type: gallery_hop, species: Li, in_plane_repeats: [3, 3],
             step_size: 0.1, weight: 2}
    """

    type: Literal["gallery_hop"] = Field(
        default="gallery_hop",
        description="Discriminator selecting this move.",
    )
    species: tuple[str, ...] = Field(
        description=(
            "Element(s) whose atoms hop.  A bare symbol (``species: Li``) "
            "is accepted.  Auto-named ``gallery_hop_<symbols>``."
        ),
    )

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: Li`` as shorthand for ``species: [Li]``."""
        return _wrap_bare_symbol(v)

    @model_validator(mode="after")
    def _species_not_in_sheets(self) -> "GalleryHopMoveSpec":
        """Sheet atoms cannot hop: they define the galleries."""
        shared = sorted(set(self.species) & set(self.host_species))
        if shared:
            raise ValueError(
                f"gallery_hop species {shared} also form the sheets "
                f"(host_species {list(self.host_species)}); only atoms in "
                "the galleries can hop."
            )
        return self

    def _build_kernel(self) -> Callable:
        return layered_moves.build_gallery_hop_kernel

    def _effective_name(self) -> str:
        if self.name is not None:
            return self.name
        return "gallery_hop_" + "_".join(self.species)

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        codes = _species_codes(
            self.species, symbol_map, "gallery_hop", "GalleryHopMoveSpec"
        )
        return {"species": codes}

    def _reject_reasons(self) -> frozenset[str]:
        # "prior": no atom of the species, or the noise left the target
        # gallery.
        return frozenset({"energy", "prior"})


class LayerSlideMoveSpec(LayeredMoveSpec):
    """Rigid in-plane translation of one sheet, or of one gallery's atoms.

    With ``species`` equal to ``host_species`` the move picks one sheet
    uniformly and translates all its atoms; this changes the stacking
    registry (AA, AB, AC) of the two galleries the sheet bounds.  With
    species not in the sheets it picks one gallery uniformly and translates
    the atoms of ``species`` it contains, e.g. Li between alpha-alpha and
    alpha-beta; an empty gallery rejects the proposal.

    The translation is ``i a / (m_a k) + j b / (m_b k)`` with ``(i, j)``
    uniform on the grid without ``(0, 0)`` and ``k = subdivisions``, plus
    in-plane Gaussian noise of ``step_size``.  ``k = 1`` gives host lattice
    vectors (intercalant hollow to hollow); ``k = 3`` adds the third-cell
    shifts between AA, AB and AC stacking.

    ::

        moves:
          - {type: layer_slide, species: C, subdivisions: 3,
             step_size: 0.02, weight: 1}
          - {type: layer_slide, species: Li, subdivisions: 1,
             step_size: 0.02, weight: 1}
    """

    type: Literal["layer_slide"] = Field(
        default="layer_slide",
        description="Discriminator selecting this move.",
    )
    species: tuple[str, ...] = Field(
        description=(
            "What slides: ``host_species`` slides one sheet; other "
            "element(s) slide their atoms in one gallery.  Mixing the two "
            "is an error.  Auto-named ``layer_slide_<symbols>``."
        ),
    )
    subdivisions: int = Field(
        default=1,
        ge=1,
        description=(
            "Grid steps per host lattice vector: 1 for host lattice "
            "translations, 3 to include the AA/AB/AC stacking shifts."
        ),
    )

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: C`` as shorthand for ``species: [C]``."""
        return _wrap_bare_symbol(v)

    @model_validator(mode="after")
    def _sheet_or_gallery(self) -> "LayerSlideMoveSpec":
        """``species`` must be the whole host, or contain none of it."""
        host = set(self.host_species)
        moving = set(self.species)
        if moving & host and moving != host:
            raise ValueError(
                f"layer_slide species {list(self.species)} mixes sheet atoms "
                f"(host_species {list(self.host_species)}) with others. Use "
                "species equal to host_species to slide one sheet, or species "
                "outside the sheets to slide the atoms of one gallery."
            )
        return self

    def _slide_host(self) -> bool:
        """True when the move slides a sheet, False for a gallery's atoms."""
        return set(self.species) == set(self.host_species)

    def _build_kernel(self) -> Callable:
        return layered_moves.build_layer_slide_kernel

    def _effective_name(self) -> str:
        if self.name is not None:
            return self.name
        return "layer_slide_" + "_".join(self.species)

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        codes = _species_codes(
            self.species, symbol_map, "layer_slide", "LayerSlideMoveSpec"
        )
        return {
            "species": codes,
            "subdivisions": self.subdivisions,
            "slide_host": self._slide_host(),
        }

    def _reject_reasons(self) -> frozenset[str]:
        # "prior": the chosen gallery holds no atom of the species.
        return frozenset({"energy", "prior"})


class StackingShiftMoveSpec(LayeredMoveSpec):
    """Change the stacking registry across one gallery by tilting c.

    Picks one gallery uniformly, moves the cell boundary into it (a change of
    representation only), and adds an in-plane shift to ``c`` with the atoms
    fixed: the periodic images above the boundary slide relative to the
    atoms below, so only that gallery's registry changes, at constant volume.
    In a two-sheet cell this is the only way to change one gallery's
    registry without touching the other.

    The shift is ``i a / (m_a k) + j b / (m_b k)`` with ``(i, j)`` in
    ``{-1, 0, 1}^2`` without ``(0, 0)``, ``k = subdivisions``, plus in-plane
    noise of ``step_size``.  ``k = 1`` changes the in-plane offset of the
    intercalants between galleries (alpha-alpha / alpha-beta); ``k = 3``
    changes the sheet stacking (AA / AB / AC).

    Requires ``cell.restrict_c_in_plane: true``: otherwise the equivalent
    cells ``c + p a + q b`` weigh the registries unequally in the prior.

    ::

        moves:
          - {type: stacking_shift, subdivisions: 1, step_size: 0.02}
          - {type: stacking_shift, subdivisions: 3, step_size: 0.02}
        cell:
          restrict_c_in_plane: true
    """

    type: Literal["stacking_shift"] = Field(
        default="stacking_shift",
        description="Discriminator selecting this move.",
    )
    subdivisions: int = Field(
        default=1,
        ge=1,
        description=(
            "Grid steps per host lattice vector: 1 for host lattice "
            "translations, 3 to include the AA/AB/AC stacking shifts.  "
            "Auto-named ``stacking_shift_<subdivisions>``."
        ),
    )

    def _build_kernel(self) -> Callable:
        return layered_moves.build_stacking_shift_kernel

    def _effective_name(self) -> str:
        if self.name is not None:
            return self.name
        return f"stacking_shift_{self.subdivisions}"

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        if n_atoms is None or cell_cfg is None:
            raise ValueError(
                "StackingShiftMoveSpec.to_descriptor() requires n_atoms and "
                "cell_cfg from the resolver."
            )
        if not cell_cfg.restrict_c_in_plane:
            raise ValueError(
                "stacking_shift requires cell.restrict_c_in_plane: true. "
                "Without it the equivalent cells c + p a + q b give the "
                "stacking registries different prior weights."
            )
        return {
            "subdivisions": self.subdivisions,
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
            "restrict_c_in_plane": True,
        }

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell"})

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})


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
        SpeciesSwapMoveSpec,
        Lattice1DMoveSpec,
        Distance1DMoveSpec,
        VolumeMoveSpec,
        ShearMoveSpec,
        StretchMoveSpec,
        AlchemicalMorphMoveSpec,
        GalleryHopMoveSpec,
        LayerSlideMoveSpec,
        StackingShiftMoveSpec,
    ],
    Field(discriminator="type"),
]
