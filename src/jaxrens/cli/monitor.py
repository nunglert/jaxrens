"""NS callback implementations for monitoring, checkpointing, and I/O.

These are plugged into the NS outer loop via the callbacks parameter.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProgressCallback:
    """Prints iteration info, acceptance rates, timing."""

    def __init__(self, info_interval: int = 100):
        self.info_interval = info_interval
        self._start_time = time.time()
        self._last_print_time = self._start_time

    def on_iteration(self, iteration: int, ns_state: dict, info: dict) -> None:
        if iteration % self.info_interval != 0 and iteration != 0:
            return

        elapsed = time.time() - self._start_time
        dt = time.time() - self._last_print_time
        self._last_print_time = time.time()

        logger.info(
            "iter=%d  Emax=%.6f  log_Z=%.4f  acc=%.2f  ss=%.4f  dt=%.1fs",
            iteration,
            float(info.get("emax", 0)),
            float(info.get("log_evidence", float("-inf"))),
            float(info.get("acceptance_rate", 0)),
            float(info.get("step_size", 0)),
            dt,
        )

    def on_finish(self, ns_state: dict) -> None:
        elapsed = time.time() - self._start_time
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

    def on_iteration(self, iteration: int, ns_state: dict, info: dict) -> None:
        emax = float(info.get("emax", 0))
        if emax > self._prev_emax and iteration > 0:
            logger.warning(
                "iter=%d: Emax increased (%.6f > %.6f)", iteration, emax, self._prev_emax
            )
        self._prev_emax = emax

    def on_finish(self, ns_state: dict) -> None:
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

    def on_iteration(self, iteration: int, ns_state: dict, info: dict) -> None:
        if iteration > 0 and iteration % self.interval == 0:
            from jaxrens.io.checkpoint import save_checkpoint

            path = self.working_dir / f"{self.prefix}.checkpoint.h5"
            save_checkpoint(path, ns_state, self.symbol_map)

    def on_finish(self, ns_state: dict) -> None:
        from jaxrens.io.checkpoint import save_checkpoint

        path = self.working_dir / f"{self.prefix}.final.checkpoint.h5"
        save_checkpoint(path, ns_state, self.symbol_map)


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

    def on_iteration(self, iteration: int, ns_state: dict, info: dict) -> None:
        if iteration % self.traj_interval == 0:
            # Build a walker dict for the dead point
            worst_idx = int(info.get("worst_idx", 0))
            dead_walker = {
                "positions": ns_state["dead_positions"][ns_state["n_dead"] - 1],
                "types": ns_state["types"][0],
                "energy": float(info.get("emax", 0)),
            }
            if ns_state.get("boxes") is not None:
                dead_walker["box"] = ns_state["boxes"][worst_idx]
            self.writer.write_dead_point(iteration, dead_walker, float(info["emax"]))

        if self.energy_logger is not None:
            self.energy_logger.write_entry(
                iteration, float(info.get("emax", 0))
            )

        if self.snapshot_interval and iteration > 0 and iteration % self.snapshot_interval == 0:
            self.writer.write_walker_snapshot(iteration, ns_state)

    def on_finish(self, ns_state: dict) -> None:
        self.writer.close()
        if self.energy_logger is not None:
            self.energy_logger.close()
