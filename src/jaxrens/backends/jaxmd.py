"""jax-md energy backend.

Wraps jax-md's analytic many-body potentials (Tersoff for covalent
solids, EAM for metals) behind the ``EnergyBackend`` interface.  All-pairs
variants only — no neighbor-list state, no allocate/update/overflow
pattern (which jaxrens deliberately replaces with bucketed kernel
compilation; see CLAUDE.md "Conventions & gotchas").

jax-md is an optional dependency.  All imports are guarded.

Dynamic cell handling
---------------------
NPT changes the unit cell every iteration.  The backend constructs the
displacement function once via ``space.periodic_general(..., fractional_coordinates=False)``
with a dummy box at init; the real cell is threaded in per call as
``energy_fn(positions, new_box=cell)``.  jax-md's smap layer forwards
the kwarg into ``periodic_general``'s displacement transform, so the
JIT cache hits on subsequent calls (verified: single trace, no
recompilation across box changes).

Multi-species support
---------------------
* Tersoff: jax-md does not yet support multi-element Tersoff
  (``tersoff(..., species=Optional[Array])`` raises ``NotImplementedError``
  when species is not None).  This backend exposes single-element
  Tersoff only and ignores the ``species`` argument from the protocol.
* EAM: ``eam_from_lammps_parameters`` reads single-element DYNAMO
  ``setfl`` files; the backend mirrors that limitation.

The ``species`` argument is therefore ignored in ``__call__``;
``atomic_numbers`` is set at construction from the element the model
was parameterised for.

MIC limitation
--------------
jax-md's ``energy.tersoff`` / ``energy.eam`` use the minimum image
convention internally and have no way to enumerate periodic images
or exclude self-image triplets.  As a consequence:

* They are silently incorrect when any cell side falls below
  ``2 * r_cutoff`` (neighbours wrap to the wrong image and are
  undercounted).
* The all-pairs Tersoff path does **not** respect extensivity for
  periodic supercells (a 2x2x2 hand-built supercell of the same
  crystal does not return ``8 * e_unit_cell``), which prevents the
  position-replication trick that the other periodic backends
  (LJ / MACE / Nequix / NeuralIL) use to work around small cells.

This backend therefore intentionally exposes no ``supercell_trafo``
knob — use it only when the unit cell is large enough that
``cell_perp_distance > 2 * r_cutoff`` for every walker (under NPT,
size the ``cell.min_volume_per_atom`` prior accordingly).
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger(__name__)

_JAXMD_AVAILABLE = False
_JAXMD_IMPORT_ERROR = ""
try:
    from jax_md import energy as _jmd_energy
    from jax_md import space as _jmd_space

    _JAXMD_AVAILABLE = True
except ImportError as exc:
    _JAXMD_IMPORT_ERROR = str(exc)


def _require_jaxmd() -> None:
    if not _JAXMD_AVAILABLE:
        raise ImportError(
            f"jax-md is required for the JaxMDBackend but not installed: "
            f"{_JAXMD_IMPORT_ERROR}"
        )


def is_available() -> bool:
    return _JAXMD_AVAILABLE


# ---------------------------------------------------------------------------
# Inline Tersoff parameter sets
# ---------------------------------------------------------------------------
#
# Tersoff '88 Si parameters, matching jax-md's bundled ``tests/data/Si.tersoff``
# fixture so users can cross-reference against upstream.  Key names
# match what ``load_lammps_tersoff_parameters`` produces (Tf suffix on
# keys that collide with Python builtins / convention).  The outer list
# is mandatory: jax-md's ``tersoff(...)`` indexes ``params[0]``.

_TERSOFF_SI_88: list[dict[str, Any]] = [
    {
        "element1": "Si",
        "element2": "Si",
        "element3": "Si",
        "mTf": 3.0,
        "gamma": 1.0,
        "lam3": 1.3258,
        "cTf": 4.8381,
        "dTf": 2.0417,
        "hTf": 0.0,
        "nTf": 22.956,
        "beta": 0.33675,
        "lam2": 1.3258,
        "B": 95.373,
        "R": 3.0,
        "D": 0.2,
        "lam1": 3.2394,
        "A": 3264.7,
    }
]

_INLINE_TERSOFF_PARAMS: dict[str, list[dict[str, Any]]] = {
    "si": _TERSOFF_SI_88,
}

# Single-element atomic-number lookup for the inline param sets.
_INLINE_ELEMENT_Z: dict[str, int] = {
    "si": 14,
}


# ---------------------------------------------------------------------------
# JaxMDBackend
# ---------------------------------------------------------------------------


class JaxMDBackend:
    """jax-md energy backend (all-pairs variants).

    Satisfies the ``EnergyBackend`` protocol via duck typing.  Returns
    ``(energy, 0, False)`` for the count/overflow slots because all-pairs
    computation has no neighbor-buffer dimension.

    Args:
        energy_fn: A jax-md energy function built by ``energy.tersoff`` /
            ``energy.eam`` over a displacement function from
            ``space.periodic_general`` or ``space.free``.  Must accept
            ``new_box=cell`` for periodic systems.
        r_cutoff: Interaction cutoff (Å) — for periodic builds, used by
            the resolver to validate supercell sizing.
        atomic_numbers: Single-element tuple ``(Z,)`` the model was
            parameterised for.  Tersoff/EAM in jax-md are
            single-species in the all-pairs path.
        potential: ``"tersoff"`` or ``"eam"`` — informational tag, used
            in logging and the repr.
        periodic: Whether the displacement function was built with
            ``space.periodic_general`` (True) or ``space.free()`` (False).
    """

    def __init__(
        self,
        energy_fn: Any,
        r_cutoff: float,
        atomic_numbers: tuple[int, ...],
        potential: str,
        periodic: bool,
    ):
        self._energy_fn = energy_fn
        self.r_cutoff = float(r_cutoff)
        self._atomic_numbers = tuple(int(z) for z in atomic_numbers)
        self.num_species = len(self._atomic_numbers)
        self.potential = potential
        self.periodic = periodic

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        """Atomic numbers (Z) the model was parameterised for.

        Used by the resolver to map user-supplied Z values in
        ``start_species`` to the model's species indices.  Tersoff and
        EAM (single-element jax-md paths) expose a single Z.
        """
        return self._atomic_numbers

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        del species, max_neighbors, ensemble_params  # all-pairs single-species
        if self.periodic:
            e = self._energy_fn(positions, new_box=cell)
        else:
            e = self._energy_fn(positions)
        return e, 0, False

    def __repr__(self) -> str:
        return (
            f"JaxMDBackend(potential={self.potential!r}, "
            f"periodic={self.periodic}, r_cutoff={self.r_cutoff}, "
            f"Z={self._atomic_numbers})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_displacement_fn(periodic: bool) -> Any:
    """Build a displacement function suitable for jax-md energies.

    Periodic case uses ``periodic_general`` with a dummy initial box
    (``eye(3)``); the real cell is supplied per call via the
    ``new_box=`` kwarg that jax-md threads through smap's ``**kwargs``.
    """
    if periodic:
        disp_fn, _ = _jmd_space.periodic_general(
            jnp.eye(3), fractional_coordinates=False,
        )
        return disp_fn
    disp_fn, _ = _jmd_space.free()
    return disp_fn


def _resolve_tersoff_params(
    tersoff_params: str | None,
    tersoff_params_file: str | None,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Resolve Tersoff parameters from either the inline name or a file.

    Returns ``(params_list, atomic_numbers)`` where ``params_list`` is
    the jax-md list-of-dicts form indexable as ``params[0]`` and
    ``atomic_numbers`` is the single-element tuple for the modelled
    species.
    """
    n_set = sum(x is not None for x in (tersoff_params, tersoff_params_file))
    if n_set != 1:
        raise ValueError(
            "Exactly one of `tersoff_params` (inline name) or "
            "`tersoff_params_file` (LAMMPS-format path) must be set."
        )

    if tersoff_params is not None:
        key = tersoff_params.lower()
        if key not in _INLINE_TERSOFF_PARAMS:
            raise ValueError(
                f"Unknown inline Tersoff parameter set {tersoff_params!r}. "
                f"Available: {sorted(_INLINE_TERSOFF_PARAMS)}.  Use "
                f"`tersoff_params_file` for arbitrary LAMMPS-format files."
            )
        return _INLINE_TERSOFF_PARAMS[key], (_INLINE_ELEMENT_Z[key],)

    with open(tersoff_params_file) as f:
        params = _jmd_energy.load_lammps_tersoff_parameters(f)

    # Single-element file: first triplet's element1/2/3 must agree.
    elem = params[0]["element1"]
    if not (elem == params[0]["element2"] == params[0]["element3"]):
        raise ValueError(
            f"Multi-element Tersoff parameter files are not supported by "
            f"jax-md's all-pairs Tersoff path (file has {elem}/"
            f"{params[0]['element2']}/{params[0]['element3']})."
        )
    z = _ELEMENT_TO_Z.get(elem)
    if z is None:
        raise ValueError(
            f"Unknown element symbol {elem!r} in Tersoff parameter file; "
            f"extend `_ELEMENT_TO_Z` in jaxmd.py."
        )
    return params, (z,)


# Minimal element → Z map covering the elements Tersoff parameter files
# in the wild target (Si, C, Ge, plus a few extras).  Extend as needed.
_ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Si": 14, "P": 15, "S": 16,
    "Ti": 22, "Cu": 29, "Ge": 32, "As": 33, "Ga": 31,
}


def create_jaxmd(
    potential: str,
    periodic: bool,
    tersoff_params: str | None = None,
    tersoff_params_file: str | None = None,
    eam_params_file: str | None = None,
) -> JaxMDBackend:
    """Create a jax-md backend.

    Args:
        potential: ``"tersoff"`` or ``"eam"``.
        periodic: Whether to build the displacement function over
            ``periodic_general`` (True; supports dynamic cell via
            ``new_box=`` kwarg) or ``free`` (False).
        tersoff_params: Inline parameter-set name (currently ``"si"``).
            Mutually exclusive with ``tersoff_params_file``.
        tersoff_params_file: Path to a LAMMPS-format Tersoff parameter
            file.  Only single-element files are supported (jax-md's
            all-pairs Tersoff is single-species).
        eam_params_file: Path to a LAMMPS DYNAMO ``setfl`` EAM file.
            Required for ``potential="eam"``.

    Returns:
        ``JaxMDBackend`` instance.
    """
    _require_jaxmd()

    disp_fn = _build_displacement_fn(periodic)

    if potential == "tersoff":
        params, atomic_numbers = _resolve_tersoff_params(
            tersoff_params, tersoff_params_file,
        )
        energy_fn = _jmd_energy.tersoff(disp_fn, params)
        r_cutoff = float(params[0]["R"]) + float(params[0]["D"])
        logger.info(
            "JaxMDBackend created: tersoff, periodic=%s, r_cutoff=%.3f, Z=%s",
            periodic, r_cutoff, atomic_numbers,
        )
        return JaxMDBackend(
            energy_fn=energy_fn,
            r_cutoff=r_cutoff,
            atomic_numbers=atomic_numbers,
            potential="tersoff",
            periodic=periodic,
        )

    if potential == "eam":
        if eam_params_file is None:
            raise ValueError(
                "`eam_params_file` is required for potential='eam'."
            )
        # ``load_lammps_eam_parameters`` consumes the whole file, so we
        # open it twice — once for the element-symbol parse, once for
        # the spline loader.
        with open(eam_params_file) as f:
            element_line = f.read().split("\n")[3].split()
        with open(eam_params_file) as f:
            charge_fn, embedding_fn, pairwise_fn, r_cutoff = (
                _jmd_energy.load_lammps_eam_parameters(f)
            )
        # DYNAMO setfl line 4: "<n_elements> <El1> <El2> ...".
        # For single-element, n_elements=1 and the symbol is the second
        # token.
        n_elements = int(element_line[0])
        if n_elements != 1:
            raise ValueError(
                f"Multi-element EAM ({n_elements} elements) is not "
                f"supported by the all-pairs jax-md path."
            )
        elem = element_line[1]
        z = _ELEMENT_TO_Z.get(elem)
        if z is None:
            raise ValueError(
                f"Unknown element symbol {elem!r} in EAM file; "
                f"extend `_ELEMENT_TO_Z` in jaxmd.py."
            )
        energy_fn = _jmd_energy.eam(
            disp_fn, charge_fn, embedding_fn, pairwise_fn,
        )
        logger.info(
            "JaxMDBackend created: eam, periodic=%s, r_cutoff=%.3f, Z=(%d,)",
            periodic, r_cutoff, z,
        )
        return JaxMDBackend(
            energy_fn=energy_fn,
            r_cutoff=float(r_cutoff),
            atomic_numbers=(z,),
            potential="eam",
            periodic=periodic,
        )

    raise ValueError(
        f"Unknown jax-md potential {potential!r}. "
        f"Supported: 'tersoff', 'eam'."
    )
