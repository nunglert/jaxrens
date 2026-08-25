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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import (
    alchemical,
    galilean,
    hmc,
    random_walk,
    shear,
    single_atom,
    stretch,
    swap,
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

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: Ge`` as shorthand for ``species: [Ge]``."""
        return (v,) if isinstance(v, str) else v

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
        if not self.species:
            return None
        if symbol_map is None:
            raise ValueError(
                "GMCMoveSpec with species=... requires symbol_map to be "
                "provided by the resolver (it maps element symbols to the "
                "type codes stored in WalkerState.types). Build descriptors "
                "via ResolvedConfig.move_descriptors rather than "
                "setup_mwg()/MoveConfig, which carry no species information."
            )
        code_of = {sym: code for code, sym in symbol_map.items()}
        unknown = [s for s in self.species if s not in code_of]
        if unknown:
            raise ValueError(
                f"gmc species {unknown} not present in the system "
                f"(symbols: {sorted(code_of)}). A species-scoped move whose "
                f"element is absent would silently become a no-op that always "
                f"accepts."
            )
        return tuple(code_of[s] for s in self.species)

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"n_reflect": self.n_reflect}
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
                lambda positions, types: jnp.zeros_like(positions),
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

    ::

        moves:
          - {type: single_atom, step_size: 0.2, weight: 1}
    """

    type: Literal["single_atom"] = Field(
        default="single_atom",
        description="Discriminator selecting this move.",
    )

    def _build_kernel(self) -> Callable:
        return single_atom.build_kernel


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
        VolumeMoveSpec,
        ShearMoveSpec,
        StretchMoveSpec,
        AlchemicalMorphMoveSpec,
    ],
    Field(discriminator="type"),
]
