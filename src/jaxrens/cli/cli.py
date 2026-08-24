"""argparse entry point for jaxrens.

Subcommands
-----------
run        Load YAML, validate, resolve, execute NS.
validate   Load YAML, validate only; print OK summary.
dump-schema  Print JSON schema for RootSpec.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from jaxrens.cli.schema.root import RootSpec

# JAX-using imports (``jaxrens.cli.{resolve,run,schema}``) are deliberately
# resolved inside the handler functions that need them.  This keeps the
# ``jaxrens plot`` and ``jaxrens dump-schema`` subcommands JAX-free so they
# run on a GPU-less or memory-pressured machine without paying for the JAX
# backend probe.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner / version reporting
# ---------------------------------------------------------------------------

# Wordmark rendered with figlet's "small" font.  Kept as a literal so the
# --help / --version paths stay dependency-free and JAX-free.
_WORDMARK = r"""  _
 (_)__ ___ ___ _ ___ _ _  ___
 | / _` \ \ / '_/ -_) ' \(_-<
_/ \__,_/_\_\_| \___|_||_/__/
|__/"""

_TAGLINE = "JAX-based nested sampling for atomistic systems"


def _package_version() -> str:
    """Return the installed jaxrens version string (JAX-free)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("jaxrens")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _banner(stream: Any = None) -> str:
    """Colourised wordmark + tagline for --help / --version headers."""
    from jaxrens.cli.style import style

    if stream is None:
        stream = sys.stdout
    mark = style(_WORDMARK, "cyan", "bold", stream=stream)
    tag = style(_TAGLINE, "dim", stream=stream)
    ver = style(f"v{_package_version()}", "green", stream=stream)
    return f"{mark}\n\n{tag}  ·  {ver}"


def _version_report() -> str:
    """Multi-line ``--version`` output: jaxrens + key runtime versions.

    Versions are read from installed package metadata, so this does *not*
    import JAX (which would pay the backend-probe cost on a GPU-less box).
    """
    import platform
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    from jaxrens.cli.style import style

    def _dep(name: str) -> str:
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            return "not installed"

    py = f"{platform.python_version()} ({platform.python_implementation()})"
    rows = [
        ("jax", _dep("jax")),
        ("jaxlib", _dep("jaxlib")),
        ("numpy", _dep("numpy")),
        ("python", py),
    ]
    width = max(len(k) for k, _ in rows)
    body = "\n".join(
        f"  {style(k.ljust(width), 'grey')}  {v}" for k, v in rows
    )
    header = style(f"jaxrens {_package_version()}", "cyan", "bold")
    return f"{header}\n{body}"


class _VersionAction(argparse.Action):
    """`--version` handler that prints the styled multi-line report."""

    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", 0)
        kwargs.setdefault("help", "Show version information and exit.")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(_version_report())
        parser.exit()


# ---------------------------------------------------------------------------
# --set dotted-path parser (≤40 lines)
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"^(\w+)\[(\d+)\]$")


def _parse_set_override(spec: str) -> tuple[list[str | int], Any]:
    """Parse ``key.path=value`` into a key-path list and a typed value.

    Supports bracket notation: ``moves[0].step_size=0.5``.

    Args:
        spec: A string of the form ``dotted.key[idx]=value``.

    Returns:
        ``(path, value)`` where *path* is a list of str/int keys and
        *value* is the YAML-parsed scalar.
    """
    if "=" not in spec:
        raise ValueError(
            f"--set expects KEY=VALUE, got {spec!r} with no '='. "
            f"Use dotted keys to reach nested config, with bracket "
            f"notation for list entries, e.g. --set run.n_live=128 or "
            f"--set moves[0].step_size=0.1."
        )
    key_str, _, val_str = spec.partition("=")
    raw_parts = key_str.split(".")
    path: list[str | int] = []
    for part in raw_parts:
        m = _BRACKET_RE.match(part)
        if m:
            path.append(m.group(1))
            path.append(int(m.group(2)))
        else:
            path.append(part)
    value = yaml.safe_load(val_str)
    return path, value


def _deep_set(d: dict[str, Any], path: list[str | int], value: Any) -> None:
    """Mutate *d* in-place, setting the leaf at *path* to *value*.

    Creates intermediate dicts when a key is missing.  List indices must
    refer to an already-existing list element.
    """
    node: Any = d
    for segment in path[:-1]:
        if isinstance(segment, int):
            node = node[segment]
        else:
            node = node.setdefault(segment, {})
    last = path[-1]
    if isinstance(last, int):
        node[last] = value
    else:
        node[last] = value


def _apply_overrides(
    raw: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """Return a copy of *raw* with all ``--set`` overrides applied."""
    import copy

    d = copy.deepcopy(raw)
    for spec in overrides:
        path, value = _parse_set_override(spec)
        _deep_set(d, path, value)
    return d


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _load_and_validate(config_path: str, overrides: list[str]) -> RootSpec:
    from jaxrens.cli.schema import RootSpec

    with open(config_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return RootSpec.model_validate(raw)


def _run_single(resolved, *, writer_mode: str = "w") -> None:
    """Execute a SingleRun NS dispatch from a SingleRun ``ResolvedConfig``."""
    from jaxrens.cli.run import run_from_config

    run_from_config(
        resolved.ns,
        list(resolved.moves),
        resolved.backend,
        resolved.output,
        initial_positions=resolved.init.initial_positions,
        initial_types=resolved.init.initial_types,
        initial_energies=resolved.init.initial_energies,
        initial_cells=resolved.init.initial_cells,
        initial_max_neighbor_counts=resolved.init.initial_max_neighbor_counts,
        symbol_map=resolved.init.symbol_map,
        restart_state=resolved.init.restart_state,
        move_descriptors=list(resolved.move_descriptors),
        constraint_descriptors=resolved.constraint_descriptors,
        initial_walk_config=resolved.initial_walk_config,
        adaptation_config=resolved.adaptation_cfg,
        termination_criteria=list(resolved.termination),
        base_backend=resolved.base_backend,
        writer_mode=writer_mode,
        ensemble_params=(
            resolved.ensemble_params_per_run[0]
            if resolved.ensemble_params_per_run
            else None
        ),
    )


def _assert_n_gpus(expected: int | None) -> int:
    """Assert that JAX exposes exactly ``expected`` local GPU devices.

    Returns ``0`` on success.  When the count mismatches, prints a multi-line
    diagnostic to stderr (likely cause: SLURM allocation, CUDA_VISIBLE_DEVICES,
    login-node execution) and returns a non-zero exit code suitable for
    ``sys.exit``.  When ``expected`` is ``None`` the check is skipped — preserves
    the original silently-use-what's-available behaviour for interactive runs.
    """
    if expected is None:
        return 0
    import os

    import jax

    devices = jax.local_devices()
    n_visible = len(devices)
    if n_visible == expected:
        return 0
    print(
        f"jaxrens run: --n-gpus={expected} but JAX sees {n_visible} local "
        f"device(s): {devices}.  Aborting before the resolver runs.\n"
        f"Likely causes:\n"
        f"  (1) SLURM downgraded the GPU allocation silently — verify with "
        f"`nvidia-smi -L` and `scontrol show job $SLURM_JOB_ID | grep -E "
        f"'Gres|Tres'`.\n"
        f"  (2) CUDA_VISIBLE_DEVICES is set in your environment "
        f"(currently: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}).\n"
        f"  (3) Running on a login node or in an `srun` allocation that did not "
        f"request `--gres=gpu:{expected}`.\n"
        f"Pass --n-gpus=$SLURM_GPUS_ON_NODE in your sbatch script (the standard "
        f"SLURM env var; not $SLURM_N_GPUS) so the job fails fast on downgrade.",
        file=sys.stderr,
    )
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    from jaxrens.cli.resolve import _apply_interval_units, resolve
    from jaxrens.cli.run import configure_file_logging

    rc = _assert_n_gpus(args.n_gpus)
    if rc != 0:
        return rc

    root = _load_and_validate(args.config, args.set)

    # Scale per-walker intervals to absolute iters before logging, so the
    # config dump and subsequent log lines show the values the runtime
    # actually uses.  The resolver still calls ``_apply_interval_units``
    # itself for direct (non-CLI) callers; here we flip
    # ``interval_units`` to ``"absolute"`` so that second pass becomes
    # an idempotent no-op (factor=1, identical re-rounding).
    root = _apply_interval_units(root)
    root = root.model_copy(update={"interval_units": "absolute"})

    # Hoist file logging before the resolver: heavy backends (NeuralIL,
    # MACE, nequix) spend many minutes in the resolver placing walkers
    # and JIT-compiling, and the resolver already emits ``logger.info``
    # progress messages.  Configuring the log handlers here means those
    # messages reach ``<prefix>.log`` and stderr at second 1 of the
    # run, instead of being dropped until ``run_*_from_config`` runs
    # the same call.
    #
    # Log files live in the *parent* of ``working_dir`` (i.e. the
    # experiment root, next to ``config.yaml`` and ``submit.slurm``)
    # rather than inside the output dir — keeps them visible from the
    # top of the experiment tree without ``cd output/``.
    root.output.working_dir.mkdir(parents=True, exist_ok=True)

    # Restart-vs-fresh decision tree.  Three start modes:
    #   * fresh: neither restart_file nor --resume → gate enforces; mode="w"
    #   * explicit restart: init.restart_file in YAML → skip gate; mode="a"
    #   * auto restart: --resume → discover ckpt; skip gate; mode="a"
    # --force is fresh-only.  All three pairs of restart triggers are
    # mutually exclusive; conflicts are rejected here, before any I/O.
    from jaxrens.cli.output_gate import (
        discover_checkpoint,
        enforce_clean_output_dir,
        snapshot_path_for_checkpoint,
        write_config_snapshot,
    )
    from jaxrens.cli.restart_validate import validate_restart_compatibility

    yaml_restart_file = root.init.restart_file
    if args.force and args.resume:
        sys.stderr.write(
            "jaxrens run: --force and --resume are mutually exclusive.\n"
        )
        return 2
    if args.force and yaml_restart_file is not None:
        sys.stderr.write(
            "jaxrens run: --force is incompatible with init.restart_file; "
            "remove restart_file from the YAML if you want a fresh start.\n"
        )
        return 2
    if args.resume and yaml_restart_file is not None:
        sys.stderr.write(
            "jaxrens run: --resume and init.restart_file are mutually "
            "exclusive (both request a restart from different sources).\n"
        )
        return 2

    restart_intent = (yaml_restart_file is not None) or args.resume
    writer_mode = "a" if restart_intent else "w"

    if args.resume:
        # Auto-discovery: pick a checkpoint and inject it into the resolved
        # init slot.  Downstream Mode-D logic in the resolver kicks in as
        # if the user had set ``init.restart_file`` explicitly.
        # All other init Modes are discarded.
        chosen = discover_checkpoint(
            root.output.working_dir,
            root.output.out_file_prefix,
        )
        root = root.model_copy(
            update={
                "init": root.init.model_copy(
                    update={
                        "restart_file": chosen,
                        "start_walker_set": None,
                        "start_species": None,
                        "start_config_file": None,
                    }
                ),
            },
        )

    if not restart_intent:
        enforce_clean_output_dir(
            root.output.working_dir,
            root.output.out_file_prefix,
            force=args.force,
        )
        # Snapshot the config so a future ``--resume`` (or explicit
        # restart_file pointing at this dir) can validate compatibility.
        write_config_snapshot(
            root.output.working_dir,
            root.output.out_file_prefix,
            root,
        )
    else:
        # Strict compatibility check against the snapshot beside the
        # checkpoint.  Refuses on immutable diffs; warns on soft diffs.
        checkpoint_path = Path(root.init.restart_file)
        validate_restart_compatibility(
            root,
            checkpoint_path=checkpoint_path,
            snapshot_path=snapshot_path_for_checkpoint(checkpoint_path),
        )

    log_dir = root.output.working_dir.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_file_logging(
        working_dir=log_dir,
        prefix=root.output.out_file_prefix,
        level=root.output.log_level,
        mode=writer_mode,
    )

    # Dump the validated config (with all pydantic defaults filled in)
    # to the log so that, after the fact, you can tell which parameters
    # came from the YAML file vs which were left to defaults.  ``mode=
    # "json"`` coerces Path / enum values into strings; YAML rendering
    # is more skim-friendly than JSON in a log file.
    logger.info(
        "Parsed configuration (validated, defaults filled):\n%s",
        yaml.safe_dump(
            root.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        ).rstrip(),
    )

    from jaxrens.cli.run import (
        run_multi_gpu_from_config,
        run_sharded_from_config,
    )
    from jaxrens.sampling.batch_descriptor import ShardedSingleRun, SingleRun

    resolved = resolve(root)

    if isinstance(resolved.batcher, SingleRun):
        pressure = resolved.ensemble_params_per_run[0].get("pressure")
        logger.info(
            "[single-run] seed=%d%s",
            resolved.ns.seed,
            f" pressure={pressure:.4g}" if pressure is not None else "",
        )
        _run_single(resolved, writer_mode=writer_mode)
    elif isinstance(resolved.batcher, ShardedSingleRun):
        pressure = resolved.ensemble_params_per_run[0].get("pressure")
        logger.info(
            "[sharded-single] seed=%d n_gpu=%d n_live=%d (K_per_gpu=%d)%s",
            resolved.ns.seed,
            resolved.batcher.n_gpu,
            resolved.ns.n_live,
            resolved.ns.n_live // resolved.batcher.n_gpu,
            f" pressure={pressure:.4g}" if pressure is not None else "",
        )
        run_sharded_from_config(resolved, writer_mode=writer_mode)
    else:
        logger.info(
            "[multi-replica] n_gpu=%d n_per_gpu=%d n_total=%d pressures=%s",
            resolved.ns.n_gpu,
            resolved.ns.n_per_gpu,
            resolved.ns.n_gpu * resolved.ns.n_per_gpu,
            ", ".join(
                f"{p.get('pressure'):.4g}"
                if p.get("pressure") is not None
                else "—"
                for p in resolved.ensemble_params_per_run
            ),
        )
        run_multi_gpu_from_config(resolved, writer_mode=writer_mode)
    return 0


def _ok_header(text: str) -> str:
    """Green ``✓ OK`` header line for the ``validate`` subcommand."""
    from jaxrens.cli.style import style

    mark = style("✓ OK", "green", "bold")
    return f"{mark} — {text}"


def _kv(label: str, value: str) -> str:
    """Render an aligned ``  <label>  <value>`` detail row (label dimmed)."""
    from jaxrens.cli.style import style

    return f"  {style(f'{label:<9}', 'cyan')} {value}"


def _cmd_validate(args: argparse.Namespace) -> int:
    root = _load_and_validate(args.config, args.set)

    if args.parse_only:
        n_moves = len(root.moves)
        move_types = ", ".join(m.move_type for m in root.moves)
        print(
            "\n".join(
                [
                    _ok_header("schema validation passed (parse-only)"),
                    _kv(
                        "run",
                        f"n_live={root.run.n_live}, "
                        f"max_iterations={root.run.max_iterations}",
                    ),
                    _kv("moves", f"{n_moves} move(s) [{move_types}]"),
                    _kv("backend", f"{root.backend.backend_type}"),
                    _kv(
                        "output",
                        f"format={root.output.format}, "
                        f"prefix={root.output.out_file_prefix}",
                    ),
                ]
            )
        )
        return 0

    from jaxrens.cli.resolve import resolve
    from jaxrens.sampling.batch_descriptor import ShardedSingleRun, SingleRun

    resolved = resolve(root)
    n_moves = len(resolved.moves)
    move_types = ", ".join(m.move_type for m in resolved.moves)
    n_atoms = int(resolved.init.initial_positions.shape[-2])

    if isinstance(resolved.batcher, SingleRun):
        topology = "SingleRun (1 replica, 1 GPU)"
    elif isinstance(resolved.batcher, ShardedSingleRun):
        topology = (
            f"ShardedSingleRun "
            f"(1 replica, sharded across {resolved.batcher.n_gpu} GPUs)"
        )
    else:
        topology = (
            f"n_gpu={resolved.ns.n_gpu} × "
            f"n_per_gpu={resolved.ns.n_per_gpu} = "
            f"{resolved.ns.n_gpu * resolved.ns.n_per_gpu} replica(s)"
        )

    print(
        "\n".join(
            [
                _ok_header("configuration valid"),
                _kv("topology", topology),
                _kv(
                    "run",
                    f"n_live={resolved.ns.n_live}, "
                    f"max_iterations={resolved.ns.max_iterations}",
                ),
                _kv("moves", f"{n_moves} move(s) [{move_types}]"),
                _kv(
                    "backend",
                    f"{resolved.backend.backend_type}, n_atoms={n_atoms}",
                ),
                _kv(
                    "output",
                    f"format={resolved.output.format}, "
                    f"prefix={resolved.output.out_file_prefix}",
                ),
            ]
        )
    )
    return 0


def _cmd_dump_schema(args: argparse.Namespace) -> int:
    # dump-schema only serialises pydantic models — it imports JAX-bound
    # modules transitively but executes no JAX op, so silence _jax_init's
    # CPU/TMPDIR runtime checks.  Must be set before the schema import below.
    import os

    os.environ["JAXRENS_SKIP_RUNTIME_CHECKS"] = "1"

    from jaxrens.cli.schema import RootSpec

    schema = RootSpec.model_json_schema()
    fmt = getattr(args, "format", "json")
    if fmt == "json":
        print(json.dumps(schema, indent=2))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    """Quick-look plotter for individual run artefacts.

    Dispatches by filename suffix (``.adaptation.h5`` / ``.re_stats.h5``
    / ``.energies``) and writes a sibling PNG by default.
    """
    from pathlib import Path

    from jaxrens.cli.plot import plot_file

    in_path = Path(args.file)
    out_path = Path(args.output) if args.output is not None else None
    try:
        written = plot_file(in_path, out_path)
    except ValueError as exc:
        print(f"jaxrens plot: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"jaxrens plot: file not found: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {written}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxrens",
        description=_banner(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action=_VersionAction,
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="<command>",
    )

    # -- run --
    p_run = sub.add_parser(
        "run", help="Run nested sampling from a YAML config."
    )
    p_run.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="FILE",
        help="Path to YAML config file.",
    )
    p_run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value (may be repeated; later wins).",
    )
    p_run.add_argument(
        "--n-gpus",
        type=int,
        default=None,
        dest="n_gpus",
        metavar="N",
        help=(
            "Assert that JAX sees exactly N local GPU devices; exit non-zero "
            "with a diagnostic on mismatch.  Typically passed from SLURM as "
            "--n-gpus $SLURM_GPUS_ON_NODE so jobs fail fast when the scheduler "
            "downgrades the GPU allocation silently."
        ),
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Delete pre-existing artifacts in working_dir matching "
            "out_file_prefix (.energies, .traj.*, .adaptation.h5, "
            ".checkpoint.h5, ...) before starting.  Without --force, the run "
            "aborts when any such file is present, to prevent silent "
            "overwrite/append corruption of prior output."
        ),
    )
    p_run.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume the run by auto-discovering a checkpoint in working_dir "
            "(prefers <prefix>.final.checkpoint.h5, falls back to "
            "<prefix>.checkpoint.h5; mtime tie-break).  Skips the output-dir "
            "gate and switches loggers to append mode.  Mutually exclusive "
            "with --force and with init.restart_file in the YAML."
        ),
    )

    # -- validate --
    p_val = sub.add_parser(
        "validate", help="Validate a YAML config without running."
    )
    p_val.add_argument("-c", "--config", required=True, metavar="FILE")
    p_val.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE"
    )
    p_val.add_argument(
        "--parse-only",
        action="store_true",
        default=False,
        help=(
            "Stop after pydantic schema validation; skip the resolver "
            "(no structure file read, no backend build, no walker placement). "
            "Fast — for catching typos / wrong field names without paying "
            "for heavy-backend initialization."
        ),
    )

    # -- dump-schema --
    p_dump = sub.add_parser(
        "dump-schema", help="Print the JSON schema for RootSpec."
    )
    p_dump.add_argument("--format", choices=["json"], default="json")

    # -- plot --
    p_plot = sub.add_parser(
        "plot",
        help=(
            "Render a quick-look PNG from a single run artefact "
            "(.adaptation.h5, .re_stats.h5, .energies).  Auto-detects by "
            "filename suffix."
        ),
    )
    p_plot.add_argument(
        "file",
        metavar="FILE",
        help=(
            "Artefact to plot.  Recognised suffixes: .adaptation.h5, "
            ".re_stats.h5, .energies."
        ),
    )
    p_plot.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="OUTPUT.png",
        help="Output PNG path.  Default: sibling <stem>.<kind>.png.",
    )

    return parser


def _iter_basemodel_types(annotation: Any) -> Any:
    """Yield every ``BaseModel`` subclass referenced in a pydantic
    annotation, peeling ``Optional[...]`` / ``List[...]`` / ``Union[...]``
    / ``Annotated[...]`` wrappers.
    """
    import typing

    from pydantic import BaseModel

    origin = typing.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield annotation
        return
    for arg in typing.get_args(annotation):
        if arg is type(None):
            continue
        yield from _iter_basemodel_types(arg)


def _collect_field_paths(
    model_cls: Any,
    prefix: str = "",
    _seen: set | None = None,
) -> list[str]:
    """Recursively collect dotted field paths from a pydantic model.

    For unions / discriminated unions every variant is descended into,
    so the resulting list contains paths reachable via any of them.
    """
    if _seen is None:
        _seen = set()
    if model_cls in _seen:
        return []
    _seen = _seen | {model_cls}

    paths: list[str] = []
    for name, finfo in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        paths.append(path)
        for sub_cls in _iter_basemodel_types(finfo.annotation):
            paths.extend(_collect_field_paths(sub_cls, path, _seen))
    return paths


def _suggest_for_extra_field(bad_key: str, parent_path: str) -> str:
    """Render a "Did you mean: ..." suffix for an extra-forbidden field.

    Two sources of candidates:
      * Exact-leaf matches — a field with the same name exists somewhere
        else in the schema (most common: user put a top-level field
        under a section).
      * Fuzzy matches against the *parent's own* fields via difflib.
    """
    import difflib

    from jaxrens.cli.schema import RootSpec

    all_paths = _collect_field_paths(RootSpec)

    exact = [p for p in all_paths if p.rsplit(".", 1)[-1] == bad_key]
    parent_fields = [
        p.rsplit(".", 1)[-1] if "." in p else p
        for p in all_paths
        if p.rsplit(".", 1)[0] == parent_path
        or (parent_path == "" and "." not in p)
    ]
    fuzzy = difflib.get_close_matches(bad_key, parent_fields, n=3, cutoff=0.6)
    fuzzy_paths = [
        (f"{parent_path}.{f}" if parent_path else f)
        for f in fuzzy
        if f != bad_key
    ]

    suggestions: list[str] = []
    for p in exact + fuzzy_paths:
        if p not in suggestions:
            suggestions.append(p)
    if not suggestions:
        return ""
    return f"  → did you mean: {', '.join(suggestions)}?"


def _format_validation_error(exc: Any, config_path: str) -> str:
    """Render a pydantic ``ValidationError`` as one line per problem.

    The default traceback exposes pydantic internals that confuse users
    who only want to know which YAML key is wrong.  We strip the noise,
    print ``<dotted.path>: <message> (got <value>)`` per error, and for
    "extra_forbidden" errors append a ``→ did you mean: ...?`` line with
    valid paths that share the leaf name plus fuzzy matches against the
    parent's own fields.
    """
    lines = [f"jaxrens: invalid configuration in {config_path!r}:"]
    for err in exc.errors():
        loc_parts = [str(p) for p in err["loc"]]
        loc = ".".join(loc_parts) or "<root>"
        msg = err.get("msg", "invalid value")
        got = err.get("input", None)
        if got is not None and not isinstance(got, (dict, list, tuple)):
            suffix = f" (got {got!r})"
        else:
            suffix = ""
        lines.append(f"  {loc}: {msg}{suffix}")

        if err.get("type") == "extra_forbidden" and loc_parts:
            bad_key = loc_parts[-1]
            parent_path = ".".join(loc_parts[:-1])
            hint = _suggest_for_extra_field(bad_key, parent_path)
            if hint:
                lines.append(hint)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point registered via ``[project.scripts]``."""
    from pydantic import ValidationError

    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "dump-schema": _cmd_dump_schema,
        "plot": _cmd_plot,
    }
    from jaxrens.cli.style import style

    def _err(msg: str) -> None:
        mark = style("✗", "red", "bold", stream=sys.stderr)
        print(f"{mark} {msg}", file=sys.stderr)

    handler = dispatch[args.command]
    try:
        sys.exit(handler(args))
    except ValidationError as exc:
        cfg = getattr(args, "config", "<unknown>")
        _err(_format_validation_error(exc, cfg))
        sys.exit(2)
    except FileNotFoundError as exc:
        _err(f"jaxrens: {exc}")
        sys.exit(2)
    except yaml.YAMLError as exc:
        _err(f"jaxrens: YAML parse error: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
