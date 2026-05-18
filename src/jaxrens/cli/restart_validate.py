"""Strict restart-compatibility validator.

Compares the snapshot YAML written by the original run against the
current run's config and refuses to start when any immutable field
differs.  Soft differences (move retuning, logging cadence, ...) are
logged as warnings.

Immutables are fields whose change would silently break the statistical
contract or produce a meaningless concatenated trace:

  * walker-population shape  (``run.n_live``, ``run.n_gpu``,
    ``run.n_per_gpu``)
  * energy function           (``backend`` subtree, including model file)
  * physics conditions        (``ensemble``, ``inter_re`` subtrees)
  * cell topology             (``cell`` subtree)

Walker-shape consistency between the checkpoint and the new config
(``n_atoms``, ``symbol_map``, cell ndim) is *not* checked here — that
falls out of the resolver's Mode-D load path and is enforced when JAX
materialises the walker arrays.

This module is only invoked from the restart branch in
:func:`jaxrens.cli.cli._cmd_run`; library callers that bypass the CLI
must do their own validation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterable

from jaxrens.cli.output_gate import read_config_snapshot

logger = logging.getLogger(__name__)


# Dotted paths into the ``root.model_dump(mode="json")`` tree.  Order
# determines the order rows appear in the refusal diff — keep the
# most-likely-to-bite-the-user fields first.
_IMMUTABLE_PATHS: tuple[str, ...] = (
    "run.n_live",
    "run.n_gpu",
    "run.n_per_gpu",
    "backend",
    "ensemble",
    "inter_re",
    "cell",
)

# Soft warnings — these are legitimately mutable across restart, but a
# silent change between segments often indicates a mistake worth flagging.
_WARN_PATHS: tuple[str, ...] = (
    "moves",
    "run.n_mcmc_steps",
    "run.max_iterations",
    "termination",
    "adaptation",
    "output.traj_interval",
    "output.snapshot_interval",
    "output.checkpoint_interval",
    "output.info_interval",
    "output.format",
    "interval_units",
)


_MISSING = object()


def _dig(tree: Any, path: str) -> Any:
    """Walk ``tree`` following a dotted ``path``; return ``_MISSING`` on miss."""
    node = tree
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return _MISSING
    return node


def _implied_n_total(tree: dict) -> int:
    """Replicate ``_derive_replica_axes`` n_total inference from a dumped config.

    Reads only the fields that participate in the replica-axis derivation:
    ``ensemble.pressure`` (when list-valued), ``inter_re.composition_targets``,
    ``inter_re.chemical_potentials``. Returns 1 for the single-replica case.
    Unknown / inconsistent shapes return ``-1`` so a downstream diff bubbles
    up clearly.
    """
    lengths: list[int] = []
    ens = _dig(tree, "ensemble")
    if isinstance(ens, dict):
        p = ens.get("pressure")
        if isinstance(p, list) and len(p) > 1:
            lengths.append(len(p))
    ire = _dig(tree, "inter_re")
    if isinstance(ire, dict):
        comp = ire.get("composition_targets")
        if isinstance(comp, list) and len(comp) > 1:
            lengths.append(len(comp))
        chem = ire.get("chemical_potentials")
        if isinstance(chem, list) and len(chem) > 1:
            lengths.append(len(chem))
    if not lengths:
        return 1
    if any(n != lengths[0] for n in lengths[1:]):
        return -1
    return lengths[0]


def _classify_diffs(
    snapshot: dict, current: dict, paths: Iterable[str]
) -> list[tuple[str, Any, Any]]:
    out: list[tuple[str, Any, Any]] = []
    for path in paths:
        a = _dig(snapshot, path)
        b = _dig(current, path)
        if a is _MISSING and b is _MISSING:
            continue
        if a != b:
            out.append((path, a, b))
    return out


def _format_value(v: Any, *, width: int = 60) -> str:
    if v is _MISSING:
        return "<missing>"
    s = repr(v)
    if len(s) > width:
        s = s[: width - 3] + "..."
    return s


def validate_restart_compatibility(
    root: Any,
    *,
    checkpoint_path: Path,
    snapshot_path: Path | None,
) -> None:
    """Refuse-or-warn on snapshot/current-config differences.

    Args:
        root: Current ``RootSpec`` (the YAML the user passed to this run).
        checkpoint_path: Path to the checkpoint being resumed (used only
            in error messages).
        snapshot_path: Path to ``{prefix}.config.snapshot.yaml`` written
            by the original run.  ``None`` (or a non-existent path) means
            the original run predates the snapshot machinery; the
            function logs a warning and returns.

    Raises:
        SystemExit: With code 2 when any immutable field differs.
    """
    if snapshot_path is None or not Path(snapshot_path).exists():
        logger.warning(
            "[restart] config snapshot not found (looked for %s); "
            "skipping snapshot-based compatibility check.  "
            "Run will proceed; consider re-running the original from a "
            "post-fix jaxrens to get full restart validation.",
            snapshot_path,
        )
        return

    snapshot = read_config_snapshot(snapshot_path)
    current = root.model_dump(mode="json")

    # Replica-count check: dedicated, fires before the broader ensemble/inter_re
    # subtree diff so the user sees a targeted message rather than a 60-char
    # truncation of the full subtree.  Any change to the implied n_total breaks
    # the checkpoint's batch shape and would crash later in the resolver, so
    # refuse here with a clearer message.
    snap_n_total = _implied_n_total(snapshot)
    cur_n_total = _implied_n_total(current)
    if snap_n_total != cur_n_total:
        sys.stderr.write(
            f"jaxrens run: restart from {checkpoint_path} is incompatible "
            f"with the current config: replica count changed.\n"
            f"  snapshot implies n_total = {snap_n_total} replicas\n"
            f"  current  implies n_total = {cur_n_total} replicas\n"
            f"  (driven by len(ensemble.pressure) / "
            f"len(inter_re.composition_targets) / "
            f"len(inter_re.chemical_potentials))\n"
            f"  snapshot: {snapshot_path}\n"
            f"Use a different working_dir for this configuration, or revert "
            f"the replica-list length.\nAborting.\n"
        )
        raise SystemExit(2)

    hard = _classify_diffs(snapshot, current, _IMMUTABLE_PATHS)
    soft = _classify_diffs(snapshot, current, _WARN_PATHS)

    if hard:
        col = max(len(p) for p, _, _ in hard)
        lines = [
            f"  {p:<{col}}  snapshot={_format_value(a)}  current={_format_value(b)}"
            for p, a, b in hard
        ]
        sys.stderr.write(
            f"jaxrens run: restart from {checkpoint_path} is incompatible "
            f"with the current config on immutable fields:\n"
            + "\n".join(lines)
            + "\n"
            f"  snapshot: {snapshot_path}\n"
            "Use a different working_dir for this configuration, or revert "
            "the conflicting fields.\nAborting.\n"
        )
        raise SystemExit(2)

    for path, snap_val, cur_val in soft:
        logger.warning(
            "[restart] %s changed across restart: snapshot=%s current=%s",
            path, _format_value(snap_val), _format_value(cur_val),
        )

    # Seed: warn when it is *unchanged* on restart — the continuation
    # otherwise uses the identical PRNG stream from this point, making
    # the new segment statistically useless as an ensemble member.
    snap_seed = _dig(snapshot, "run.seed")
    cur_seed = _dig(current, "run.seed")
    if snap_seed is not _MISSING and snap_seed == cur_seed:
        logger.warning(
            "[restart] run.seed (%s) is unchanged from the original run.  "
            "Bump the seed if you intend the restart as an independent "
            "ensemble member.",
            snap_seed,
        )
