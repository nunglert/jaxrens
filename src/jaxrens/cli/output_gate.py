"""Output-dir safety gate and restart-checkpoint discovery for ``jaxrens run``.

The gate refuses to start when ``working_dir`` already holds artifacts
matching the configured ``out_file_prefix`` — without it, re-running the
same config against the same output dir silently truncates ``.energies``
and concatenates duplicate frames into ``.traj.extxyz``.

The same prefix-aware globbing is reused by :func:`discover_checkpoint`
for the ``--resume`` (auto-restart) flow, which looks up a checkpoint in
``working_dir`` instead of taking an explicit path from the YAML.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artefact catalogue
# ---------------------------------------------------------------------------

# Glob patterns matched against ``working_dir`` to decide whether a
# fresh-run gate should refuse.  Keep in sync with every writer/callback
# that emits files into ``working_dir``.
_ARTIFACT_GLOBS: tuple[str, ...] = (
    "{prefix}.energies",
    "{prefix}.run*.energies",
    "{prefix}.traj.*",
    "{prefix}.run*.traj.*",
    "{prefix}.adaptation.h5",
    "{prefix}.re_stats.h5",
    "{prefix}.max_neighbors.h5",
    "{prefix}.acc_rates.h5",
    "{prefix}.checkpoint.h5",
    "{prefix}.initial.checkpoint.h5",
    "{prefix}.final.checkpoint.h5",
    "{prefix}.config.snapshot.yaml",
)

# Suffixes recognised as restart sources by ``discover_checkpoint``.
# Ordered by *preference* on mtime ties: a clean final checkpoint beats
# a rolling one written at the same instant.
_CHECKPOINT_SUFFIXES: tuple[str, ...] = (
    ".final.checkpoint.h5",
    ".checkpoint.h5",
)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _find_artifacts(working_dir: Path, prefix: str) -> list[Path]:
    if not working_dir.is_dir():
        return []
    found: set[Path] = set()
    for pattern in _ARTIFACT_GLOBS:
        for hit in working_dir.glob(pattern.format(prefix=prefix)):
            if hit.is_file():
                found.add(hit)
    return sorted(found)


def enforce_clean_output_dir(
    working_dir: Path | str, prefix: str, *, force: bool
) -> None:
    """Abort the run (or wipe artifacts) if ``working_dir`` is dirty.

    Args:
        working_dir: Directory NS output files will be written to.
        prefix:      ``out_file_prefix`` from the config.
        force:       When True, delete the offending files; when False,
                     raise ``SystemExit(2)`` listing them.

    Raises:
        SystemExit: With code 2 when artifacts exist and ``force`` is False.
    """
    working_dir = Path(working_dir)
    artifacts = _find_artifacts(working_dir, prefix)
    if not artifacts:
        return

    if not force:
        shown = artifacts[:10]
        more = len(artifacts) - len(shown)
        listing = "\n".join(f"  {p.name}" for p in shown)
        if more > 0:
            listing += f"\n  ... +{more} more"
        sys.stderr.write(
            f"jaxrens run: output directory already contains artifacts for "
            f"prefix {prefix!r}:\n{listing}\n"
            f"  in: {working_dir}\n"
            f"Pass --force to delete them and start fresh.\n"
        )
        raise SystemExit(2)

    for path in artifacts:
        path.unlink()


# ---------------------------------------------------------------------------
# Config snapshot — written on fresh start, consumed by the restart validator
# ---------------------------------------------------------------------------


def snapshot_filename(prefix: str) -> str:
    """The fixed filename a fresh ``jaxrens run`` writes its config to."""
    return f"{prefix}.config.snapshot.yaml"


def write_config_snapshot(
    working_dir: Path | str, prefix: str, root: Any
) -> Path:
    """Dump the validated root config to ``{prefix}.config.snapshot.yaml``.

    Called once on a fresh run, after the gate has cleared and before any
    writer instantiates.  On restart, the validator reads this file back
    to diff against the resumed run's config.
    """
    import yaml

    working_dir = Path(working_dir)
    path = working_dir / snapshot_filename(prefix)
    payload = root.model_dump(mode="json")
    with open(path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    return path


def read_config_snapshot(path: Path | str) -> dict:
    """Load a previously-written config snapshot YAML into a dict."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def snapshot_path_for_checkpoint(checkpoint_path: Path | str) -> Path | None:
    """Locate the snapshot YAML that *should* sit next to a checkpoint.

    Strips ``.checkpoint.h5`` / ``.final.checkpoint.h5`` from the checkpoint
    filename to recover the prefix, then forms
    ``<ckpt_dir>/<prefix>.config.snapshot.yaml``.  Returns ``None`` if the
    checkpoint filename does not match the recognised suffixes — the caller
    can decide whether to error or skip snapshot-based validation.
    """
    checkpoint_path = Path(checkpoint_path)
    name = checkpoint_path.name
    for suffix in _CHECKPOINT_SUFFIXES:
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            return checkpoint_path.parent / snapshot_filename(prefix)
    return None


# ---------------------------------------------------------------------------
# Checkpoint discovery — ``--resume`` (auto-restart)
# ---------------------------------------------------------------------------


def discover_checkpoint(working_dir: Path | str, prefix: str) -> Path:
    """Pick the checkpoint to restart from in ``working_dir``.

    Strategy:
      * Look for ``{prefix}.final.checkpoint.h5`` and ``{prefix}.checkpoint.h5``.
      * If neither exists → ``SystemExit(2)`` listing the dir.
      * If exactly one exists → use it.
      * If both exist → use the higher-mtime one.  On exact mtime tie the
        ``.final`` variant wins (a completed-run final beats the
        last rolling write that produced it).
      * The chosen path is logged so the audit trail records which file
        actually drove the restart.

    Returns:
        Absolute path of the chosen checkpoint.

    Raises:
        SystemExit: With code 2 when no candidate exists.
    """
    working_dir = Path(working_dir)
    candidates: list[tuple[Path, float, int]] = []
    for rank, suffix in enumerate(_CHECKPOINT_SUFFIXES):
        path = working_dir / f"{prefix}{suffix}"
        if path.is_file():
            candidates.append((path, path.stat().st_mtime, rank))

    if not candidates:
        sys.stderr.write(
            f"jaxrens run --resume: no checkpoint matching prefix "
            f"{prefix!r} in {working_dir}.\n"
            f"  Expected one of: "
            f"{', '.join(f'{prefix}{s}' for s in _CHECKPOINT_SUFFIXES)}\n"
        )
        raise SystemExit(2)

    # Sort: highest mtime first, then lowest rank (``final`` preferred) on tie.
    candidates.sort(key=lambda t: (-t[1], t[2]))
    chosen, mtime, _ = candidates[0]
    if len(candidates) > 1:
        other, other_mtime, _ = candidates[1]
        logger.info(
            "[--resume] selected %s (mtime=%.3f); also found %s (mtime=%.3f)",
            chosen.name, mtime, other.name, other_mtime,
        )
    else:
        logger.info("[--resume] selected %s (mtime=%.3f)", chosen.name, mtime)

    return chosen
