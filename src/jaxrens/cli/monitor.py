"""NS callback implementations for monitoring, checkpointing, and I/O.

These are plugged into the NS outer loop via the callbacks parameter.
Callbacks receive NSState objects (not dicts).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from jaxrens.state.ns import NSState
from jaxrens.utils.cell import get_volume

logger = logging.getLogger(__name__)


def _ns_state_to_checkpoint_dict(ns_state: NSState) -> dict:
    """Convert NSState to a dict suitable for save_checkpoint."""
    pop = ns_state.population
    ep = ns_state.population.ensemble_params if hasattr(ns_state.population, "ensemble_params") else {}
    is_npt = isinstance(ep, dict) and float(ep.get("pressure", 0.0)) != 0.0
    result = {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "dead_energies": ns_state.dead_energies,
        "dead_positions": ns_state.dead_positions,
        "dead_volumes": ns_state.dead_volumes if is_npt else None,
        "log_evidence": ns_state.log_evidence,
        "iteration": int(ns_state.iteration),
        "n_dead": int(ns_state.n_dead),
        "n_walkers": ns_state.n_walkers,
    }
    if is_npt:
        result["live_volumes"] = jax.vmap(get_volume)(pop.cell)
    else:
        result["live_volumes"] = None
    return result


class ProgressCallback:
    """Prints iteration info, acceptance rates, timing."""

    def __init__(self, info_interval: int = 100):
        self.info_interval = info_interval
        self._start_time = time.time()
        self._last_print_time = self._start_time

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration % self.info_interval != 0 and iteration != 0:
            return

        elapsed = time.time() - self._start_time
        dt = time.time() - self._last_print_time
        self._last_print_time = time.time()

        logger.info(
            "iter=%d  Emax=%.6f  log_Z=%.4f  acc=%.2f  dt=%.1fs",
            iteration,
            float(info.get("emax", 0)),
            float(info.get("acceptance_rate", 0)),
            float(ns_state.log_evidence) if isinstance(ns_state, NSState) else float(info.get("log_evidence", float("-inf"))),
            dt,
        )

    def on_finish(self, ns_state: Any) -> None:
        elapsed = time.time() - self._start_time
        if isinstance(ns_state, NSState):
            logger.info(
                "NS finished: %d iterations, log_Z=%.4f, elapsed=%.1fs",
                int(ns_state.iteration),
                float(ns_state.log_evidence),
                elapsed,
            )
        else:
            logger.info(
                "NS finished: %d iterations, log_Z=%.4f, elapsed=%.1fs",
                ns_state["iteration"],
                float(ns_state["log_evidence"]),
                elapsed,
            )


class EnergyCheckCallback:
    """Warns if energy is not decreasing as expected."""

    def __init__(self):
        self._prev_emax = float("inf")

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        emax = float(info.get("emax", 0))
        if emax > self._prev_emax and iteration > 0:
            logger.warning(
                "iter=%d: Emax increased (%.6f > %.6f)", iteration, emax, self._prev_emax
            )
        self._prev_emax = emax

    def on_finish(self, ns_state: Any) -> None:
        pass


class CheckpointCallback:
    """Saves checkpoints at configured intervals."""

    def __init__(
        self,
        working_dir: Path | str,
        interval: int = 100,
        prefix: str = "ns",
        symbol_map: dict[int, str] | None = None,
    ):
        self.working_dir = Path(working_dir)
        self.interval = interval
        self.prefix = prefix
        self.symbol_map = symbol_map

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration > 0 and iteration % self.interval == 0:
            from jaxrens.io.checkpoint import save_checkpoint

            path = self.working_dir / f"{self.prefix}.checkpoint.h5"
            state_dict = _ns_state_to_checkpoint_dict(ns_state) if isinstance(ns_state, NSState) else ns_state
            save_checkpoint(path, state_dict, self.symbol_map)

    def on_finish(self, ns_state: Any) -> None:
        from jaxrens.io.checkpoint import save_checkpoint

        path = self.working_dir / f"{self.prefix}.final.checkpoint.h5"
        state_dict = _ns_state_to_checkpoint_dict(ns_state) if isinstance(ns_state, NSState) else ns_state
        save_checkpoint(path, state_dict, self.symbol_map)


class TrajectoryCallback:
    """Writes dead points and snapshots at configured intervals."""

    def __init__(
        self,
        writer: Any,
        energy_logger: Any | None = None,
        traj_interval: int = 1,
        snapshot_interval: int = 100,
    ):
        self.writer = writer
        self.energy_logger = energy_logger
        self.traj_interval = traj_interval
        self.snapshot_interval = snapshot_interval

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration % self.traj_interval == 0:
            if isinstance(ns_state, NSState):
                dead_walker = {
                    "positions": ns_state.dead_positions[ns_state.n_dead - 1],
                    "types": ns_state.population.types[0],
                    "energy": float(info.get("emax", 0)),
                }
                # cell is always present on MCState (zeros for non-periodic)
                worst_idx = int(info.get("worst_idx", 0))
                cell = ns_state.population.cell[worst_idx]
                if jnp.any(cell != 0):
                    dead_walker["box"] = cell
            else:
                worst_idx = int(info.get("worst_idx", 0))
                dead_walker = {
                    "positions": ns_state["dead_positions"][ns_state["n_dead"] - 1],
                    "types": ns_state["types"][0],
                    "energy": float(info.get("emax", 0)),
                }
                if ns_state.get("cells") is not None:
                    dead_walker["box"] = ns_state["cells"][worst_idx]
            self.writer.write_dead_point(iteration, dead_walker, float(info["emax"]))

        if self.energy_logger is not None:
            self.energy_logger.write_entry(
                iteration, float(info.get("emax", 0))
            )

        if self.snapshot_interval and iteration > 0 and iteration % self.snapshot_interval == 0:
            if isinstance(ns_state, NSState):
                snapshot_dict = _ns_state_to_checkpoint_dict(ns_state)
                self.writer.write_walker_snapshot(iteration, snapshot_dict)
            else:
                self.writer.write_walker_snapshot(iteration, ns_state)

    def on_finish(self, ns_state: Any) -> None:
        self.writer.close()
        if self.energy_logger is not None:
            self.energy_logger.close()
