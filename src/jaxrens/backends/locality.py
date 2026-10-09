"""Locality capability declaration and probing for EnergyBackend.

A backend's ``r_cutoff`` being finite does NOT by itself imply that a small
subset of the energy can be recomputed cheaply and exactly: message-passing
GNNs (MACE, nequix) have a per-layer ``r_cutoff`` but an *effective*
receptive field of ``r_cutoff * n_layers`` — a naive single-hop subset
re-evaluation under their raw ``r_cutoff`` would silently omit second-hop
contributions and produce a wrong-but-non-crashing energy. Locality is
therefore a capability a backend must explicitly assert, mirroring the
existing duck-typed optional methods on :class:`~jaxrens.backends.base.EnergyBackend`
(``energy_and_forces``, ``members``, ``max_neighbors_for``) rather than
something inferred from ``r_cutoff`` alone.

Only :class:`~jaxrens.backends.neuralil.NeuralILBackend` declares a
``"single_cutoff"`` capability today — it is a strictly local,
single-cutoff-radius descriptor model (Behler-Parrinello style), not
multi-layer message passing. MACE/nequix (extended receptive field), LJ
(already O(N^2), no real neighbor list to exploit), and the toy backends
(``harmonic``/``double_well``/``gaussian_mixture``, not per-atom-additive at
all in the ``gaussian_mixture`` case) deliberately do NOT declare a
capability — a future contributor adding one for a message-passing backend
must account for the larger effective cutoff, not reuse the single-hop
implementation built for NeuralIL.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class LocalEnergyCapability:
    """Declares how a backend's energy decomposes over local neighborhoods.

    Attributes:
        kind: One of:
            - ``"single_cutoff"``: the total energy is an exact sum of
              per-atom terms, each depending only on atoms within
              ``effective_cutoff`` of the center.
            - ``"extended"``: per-atom-decomposable in principle, but the
              effective receptive field is larger than any single
              ``r_cutoff`` the backend exposes (multi-layer message-passing
              GNNs). Reserved for future use — no backend declares this yet.
            - ``"none"``: not meaningfully per-atom-decomposable, or no
              subset-energy method exists. No local update is possible.
        effective_cutoff: Radius such that any atom farther than this from
            every changed atom has a provably unchanged per-atom energy
            contribution. Meaningless (0.0) for ``"none"``.
        supports_subset_query: Whether the backend implements
            ``atomic_energies_for`` (see :mod:`jaxrens.backends.neuralil`).
    """

    kind: str
    effective_cutoff: float = 0.0
    supports_subset_query: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("single_cutoff", "extended", "none"):
            raise ValueError(
                f"Unknown LocalEnergyCapability.kind={self.kind!r}; "
                f"expected 'single_cutoff', 'extended', or 'none'."
            )


_NONE_CAPABILITY = LocalEnergyCapability(kind="none")


def probe_local_capability(backend: Any) -> LocalEnergyCapability:
    """Duck-typed capability probe — the canonical way to ask "is this
    backend local-update-capable", mirroring
    ``getattr(backend, "energy_and_forces", None)`` elsewhere in the codebase.

    Returns ``LocalEnergyCapability(kind="none")`` for any backend that does
    not declare ``local_energy_capability`` (the safe default for every
    backend not explicitly updated), and also if a backend declares
    ``kind="single_cutoff"`` but does not actually implement
    ``atomic_energies_for`` — a declared-but-unimplemented capability is
    treated as unusable rather than raising, so a partially-migrated backend
    degrades to full recomputation instead of crashing.
    """
    cap = getattr(backend, "local_energy_capability", None)
    if cap is None:
        return _NONE_CAPABILITY
    if cap.kind == "single_cutoff" and not hasattr(
        backend, "atomic_energies_for"
    ):
        return _NONE_CAPABILITY
    return cap
