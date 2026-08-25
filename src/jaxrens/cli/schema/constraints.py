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
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from jaxrens.constraints.base import ConstraintDescriptor
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


ConstraintSpec = Annotated[
    Union[MinDistanceConstraintSpec],
    Field(discriminator="type"),
]
