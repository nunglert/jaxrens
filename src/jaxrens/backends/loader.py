"""Unified backend loader.

load_backend(backend_type, **kwargs) -> (energy_fn, params)

Single entry point for all energy backends.
"""

from typing import Any

from jaxrens.backends.protocols import EnergyFn
from jaxrens.types import Params


def load_backend(
    backend_type: str,
    **kwargs: Any,
) -> tuple[EnergyFn, Params]:
    """Load any energy backend through a single interface.

    Returns (energy_fn, params) where params is an opaque pytree.

    Args:
        backend_type: One of "harmonic", "double_well", "gaussian_mixture",
            "lj", "neuralil", "mace".
        **kwargs: Backend-specific keyword arguments.

    Returns:
        (energy_fn, params) tuple.

    Raises:
        ValueError: If backend_type is unknown.
    """
    match backend_type:
        case "harmonic":
            from jaxrens.backends.toy import create_harmonic

            return create_harmonic(**kwargs)
        case "double_well":
            from jaxrens.backends.toy import create_double_well

            return create_double_well(**kwargs)
        case "gaussian_mixture":
            from jaxrens.backends.toy import create_gaussian_mixture

            return create_gaussian_mixture(**kwargs)
        case "lj":
            from jaxrens.backends.lj import create_lj

            return create_lj(**kwargs)
        case "neuralil":
            from jaxrens.backends.neuralil import create_neuralil

            return create_neuralil(**kwargs)
        case "mace":
            from jaxrens.backends.mace import create_mace

            return create_mace(**kwargs)
        case _:
            raise ValueError(
                f"Unknown backend: {backend_type!r}. "
                f"Available: harmonic, double_well, gaussian_mixture, lj, neuralil, mace"
            )
