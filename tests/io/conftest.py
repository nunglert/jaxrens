"""Shared fixtures for the I/O tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.io.checkpoint import save_checkpoint
from jaxrens.io.energy_log import EnergyLogger


@pytest.fixture
def minimal_run_dir(tmp_path):
    """Write ns.final.checkpoint.h5 and ns.energies into tmp_path, the
    files Monitor.from_directory needs, from a hand-built state dict.

    Tests add their own trace files (.adaptation.h5, .re_stats.h5, ...)
    next to them under the same "ns" prefix.
    """
    rng = np.random.default_rng(7)
    n_live, n_dead, n_atoms = 5, 10, 2
    dead_e = np.sort(rng.uniform(0, 5, n_dead))
    ns_state = {
        "positions": jnp.asarray(rng.uniform(0, 1, (n_live, n_atoms, 3))),
        "types": jnp.zeros(n_atoms, dtype=jnp.int32),
        "energies": jnp.asarray(rng.uniform(4, 6, n_live)),
        "cells": None,
        "log_evidence": 10.0,
        "iteration": n_dead,
        "n_dead": n_dead,
        "n_walkers": n_live,
    }
    save_checkpoint(tmp_path / "ns.final.checkpoint.h5", ns_state)

    energy_logger = EnergyLogger(tmp_path / "ns.energies", n_walkers=n_live)
    energy_logger.write_header()
    for i, e in enumerate(dead_e):
        energy_logger.write_entry(i, float(e))
    energy_logger.close()
    return tmp_path
