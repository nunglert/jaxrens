"""Pydantic schema for the ``constraints`` section of a jaxrens YAML config.

Configuration constraints are hard predicates on a walker geometry that
restrict the prior the chain explores (see :mod:`jaxrens.constraints`). Each
concrete spec validates user input and produces a
:class:`jaxrens.constraints.base.ConstraintDescriptor` via ``to_descriptor``,
which the resolver registers with ``build_mwg``.

Example YAML::

    constraints:
      - type: minimum_distance
        d_min: 0.8                 # uniform floor, Angstrom
      - type: minimum_distance
        d_min:                     # per-species-pair floors
          default: 1.0
          Si-Si: 2.0
          Si-O: 1.6
      - type: in_plane_strain      # layered cells: strain of (a, b)
        reference: [7.386, 7.386, 120.0]
        lambda_min: 0.94
        lambda_max: 1.06
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jaxrens.constraints.base import ConstraintDescriptor
from jaxrens.constraints.in_plane_strain import (
    in_plane_strain_descriptor,
    reference_lattice_2d,
)
from jaxrens.constraints.min_distance import min_distance_descriptor


class BaseConstraintSpec(BaseModel):
    """Fields shared by every constraint type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_descriptor(
        self, *, symbol_map: dict[int, str]
    ) -> ConstraintDescriptor:
        raise NotImplementedError


class MinDistanceConstraintSpec(BaseConstraintSpec):
    """Reject configurations with any inter-atomic distance below ``d_min``.

    ``d_min`` is either a single float (a uniform floor for all pairs) or a
    mapping of ``"A-B"`` element-symbol pairs to floats, with an optional
    ``"default"`` key for unspecified pairs (default 0.0 — no constraint).
    Distances use the minimum-image convention under a periodic cell.

    Unlike ``backend.softcore_repulsion``, which deforms the potential below
    its cutoff and so changes the sampled distribution, a constraint
    restricts the prior — exactly like the likelihood threshold — and costs
    nothing where it is not violated.

    ::

        constraints:
          - type: minimum_distance
            d_min: 0.8

    ::

        constraints:
          - type: minimum_distance
            d_min:
              default: 1.0
              Si-Si: 2.0
              Si-O: 1.6
    """

    type: Literal["minimum_distance"] = Field(
        default="minimum_distance",
        description="Discriminator selecting this constraint.",
    )
    d_min: float | dict[str, float] = Field(
        description=(
            "Minimum allowed interatomic distance (Angstrom).  Either a "
            "single float applied to every pair, or a mapping of "
            '``"A-B"`` element-symbol pairs to floats with an optional '
            '``"default"`` key for unlisted pairs (itself defaulting to '
            "0.0, i.e. no floor).  Distances use the minimum-image "
            "convention under a periodic cell."
        ),
    )

    def _pair_lookup(self, mapping: dict[str, float]) -> tuple[dict, float]:
        """Normalize the ``"A-B"`` mapping into a symmetric symbol-pair dict."""
        default = 0.0
        pairs: dict[frozenset[str], float] = {}
        for key, value in mapping.items():
            if key == "default":
                default = float(value)
                continue
            parts = key.split("-")
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"minimum_distance d_min key {key!r} must be a 'A-B' "
                    f"element-symbol pair (or 'default')."
                )
            pairs[frozenset(parts)] = float(value)
        return pairs, default

    def to_descriptor(
        self, *, symbol_map: dict[int, str]
    ) -> ConstraintDescriptor:
        n_types = len(symbol_map)
        matrix = np.zeros((n_types, n_types), dtype=np.float32)

        if isinstance(self.d_min, dict):
            pairs, default = self._pair_lookup(self.d_min)
            known_symbols = set(symbol_map.values())
            referenced = {s for pair in pairs for s in pair}
            unknown = referenced - known_symbols
            if unknown:
                raise ValueError(
                    f"minimum_distance d_min references element(s) "
                    f"{sorted(unknown)} not present in the system "
                    f"(symbols: {sorted(known_symbols)})."
                )
            for i in range(n_types):
                for j in range(n_types):
                    key = frozenset({symbol_map[i], symbol_map[j]})
                    matrix[i, j] = pairs.get(key, default)
        else:
            if self.d_min < 0.0:
                raise ValueError(
                    f"minimum_distance d_min must be >= 0, got {self.d_min}."
                )
            matrix[:, :] = float(self.d_min)

        # A move that only relabels species can change validity only when the
        # thresholds actually differ across pairs; flag type-dependence so
        # such moves are gated exactly when (and only when) necessary.
        type_dependent = bool(matrix.min() != matrix.max())
        return min_distance_descriptor(matrix, type_dependent=type_dependent)


class InPlaneStrainConstraintSpec(BaseConstraintSpec):
    """Bound the strain of the (a, b) lattice with respect to a reference.

    For layered cells whose sheets only move with the cell: the volume,
    shear and stretch moves then strain the sheets themselves, and this
    keeps that strain physical.  The principal stretches of (a, b) with
    respect to ``reference`` (1 = undeformed) must lie in
    ``[lambda_min, lambda_max]``.  Rigid rotations and changes of ``c`` do
    not count.  See :mod:`jaxrens.constraints.in_plane_strain`.

    ::

        constraints:
          - type: in_plane_strain
            reference: [7.386, 7.386, 120.0]   # |a|, |b| (A), angle (deg)
            lambda_min: 0.94
            lambda_max: 1.06
    """

    type: Literal["in_plane_strain"] = Field(
        default="in_plane_strain",
        description="Discriminator selecting this constraint.",
    )
    reference: tuple[float, float, float] = Field(
        description=(
            "Reference in-plane lattice as ``[|a|, |b|, gamma]``: lengths "
            "of a and b (Angstrom) and the angle between them (degrees).  "
            "Usually the (a, b) of the starting cells."
        ),
    )
    lambda_min: float = Field(
        description=("Smallest allowed principal stretch of (a, b), below 1."),
    )
    lambda_max: float = Field(
        description="Largest allowed principal stretch of (a, b), above 1.",
    )

    @model_validator(mode="after")
    def _check(self) -> "InPlaneStrainConstraintSpec":
        a, b, gamma = self.reference
        if a <= 0.0 or b <= 0.0 or not 0.0 < gamma < 180.0:
            raise ValueError(
                "in_plane_strain reference must be [|a| > 0, |b| > 0, "
                f"0 < gamma < 180], got {list(self.reference)}."
            )
        if not 0.0 < self.lambda_min < 1.0 < self.lambda_max:
            raise ValueError(
                "in_plane_strain needs 0 < lambda_min < 1 < lambda_max, got "
                f"lambda_min={self.lambda_min}, "
                f"lambda_max={self.lambda_max}."
            )
        return self

    def to_descriptor(
        self, *, symbol_map: dict[int, str]
    ) -> ConstraintDescriptor:
        return in_plane_strain_descriptor(
            reference_lattice_2d(*self.reference),
            self.lambda_min,
            self.lambda_max,
        )


ConstraintSpec = Annotated[
    Union[MinDistanceConstraintSpec, InPlaneStrainConstraintSpec],
    Field(discriminator="type"),
]
