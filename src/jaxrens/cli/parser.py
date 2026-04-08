"""Configuration file parsing.

Reads jaxnest-style ns.inp files (key=value format) and produces
frozen dataclass configs. Also supports creating configs from dicts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jaxrens.state.config import BackendConfig, MoveConfig, NSConfig, OutputConfig


def parse_input_file(path: Path | str) -> dict[str, str]:
    """Parse a key=value input file into a raw dict.

    Handles:
    - Comments (#)
    - Blank lines
    - Whitespace around = sign
    - Inline comments

    Args:
        path: Path to ns.inp file.

    Returns:
        Dict of string key-value pairs.
    """
    path = Path(path)
    result = {}
    pattern = re.compile(r"^\s*(\S+)\s*=\s*(.*\S)\s*$")

    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                result[m.group(1)] = m.group(2)

    return result


def _parse_int_list(s: str) -> list[int]:
    """Parse a space-separated string of ints."""
    return [int(x) for x in s.strip().split()]


def raw_to_configs(
    raw: dict[str, str],
) -> tuple[NSConfig, MoveConfig, BackendConfig, OutputConfig]:
    """Convert raw key-value dict to typed config dataclasses.

    Args:
        raw: Dict from parse_input_file or user-supplied dict.

    Returns:
        (NSConfig, MoveConfig, BackendConfig, OutputConfig) tuple.
    """
    pressure_str = raw.get("pressure", None)
    pressure = float(pressure_str) if pressure_str is not None else None

    ns = NSConfig(
        n_live=int(raw.get("n_walkers", "500")),
        max_iterations=int(raw.get("max_iterations", "50000")),
        convergence_threshold=float(raw.get("convergence_threshold", "0.1")),
        n_mcmc_steps=int(raw.get("n_mcmc_steps", raw.get("atom_traj_len", "20"))),
        n_cull=int(raw.get("n_cull", "1")),
        seed=int(raw.get("seed", "42")),
        platform=raw.get("platform", "gpu"),
        n_runs=int(raw.get("n_runs", raw.get("n_calc_batch", "1"))),
        pressure=pressure,
    )

    move = MoveConfig(
        move_type=raw.get("move_type", raw.get("mc_move_type", "galilean")),
        step_size=float(raw.get("step_size", raw.get("initial_step_size", "0.1"))),
        n_steps=int(raw.get("n_steps", raw.get("atom_traj_len", "10"))),
        adaptation_enabled=raw.get("adaptation_enabled", "true").lower() == "true",
        adaptation_warmup=int(raw.get("adaptation_warmup", "100")),
        target_acceptance=float(raw.get("target_acceptance", "0.5")),
    )

    max_neighbors_str = raw.get("max_neighbors_list", "30 35 40 45 50")
    backend = BackendConfig(
        backend_type=raw.get("backend_type", raw.get("backend", "lj")),
        checkpoint_path=raw.get("checkpoint_path", raw.get("pickle_file", None)),
        n_atoms=int(raw.get("n_atoms", "13")),
        periodic=raw.get("periodic", "false").lower() == "true",
        cutoff=float(raw["cutoff"]) if "cutoff" in raw else None,
        max_neighbors_list=_parse_int_list(max_neighbors_str),
        max_neighbors_offset=int(raw.get("max_neighbors_offset", "5")),
    )

    output = OutputConfig(
        format=raw.get("config_file_format", raw.get("output_format", "extxyz")),
        traj_interval=int(raw.get("traj_interval", "1")),
        snapshot_interval=int(raw.get("snapshot_interval", "100")),
        snapshot_clean=raw.get("snapshot_clean", "true").lower() == "true",
        checkpoint_interval=int(raw.get("checkpoint_interval", "100")),
        checkpoint_keep=int(raw.get("checkpoint_keep", "3")),
        info_interval=int(raw.get("info_interval", "100")),
        out_file_prefix=raw.get("out_file_prefix", "ns"),
        working_dir=Path(raw.get("working_dir", ".")),
        write_traj_db=raw.get("write_traj_db", "false").lower() == "true",
    )

    return ns, move, backend, output


def load_config(
    path: Path | str,
) -> tuple[NSConfig, MoveConfig, BackendConfig, OutputConfig]:
    """Load configuration from an ns.inp file.

    Args:
        path: Path to configuration file.

    Returns:
        (NSConfig, MoveConfig, BackendConfig, OutputConfig) tuple.
    """
    raw = parse_input_file(path)
    return raw_to_configs(raw)


def dict_to_input_str(config_dict: dict[str, Any]) -> str:
    """Convert a dict back to ns.inp format string."""
    lines = []
    for key, val in config_dict.items():
        lines.append(f"{key} = {val}")
    return "\n".join(lines)
