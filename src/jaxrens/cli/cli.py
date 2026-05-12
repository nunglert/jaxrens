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
from typing import Any

import yaml

from jaxrens.cli.resolve import _apply_interval_units, resolve
from jaxrens.cli.run import configure_file_logging, run_from_config
from jaxrens.cli.schema import RootSpec

logger = logging.getLogger(__name__)


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
        raise ValueError(f"--set value must contain '=': {spec!r}")
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


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
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
    with open(config_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return RootSpec.model_validate(raw)


def _run_single(resolved) -> None:
    """Execute a SingleRun NS dispatch from a SingleRun ``ResolvedConfig``."""
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
        initial_walk_config=resolved.initial_walk_config,
        adaptation_config=resolved.adaptation_cfg,
        termination_criteria=list(resolved.termination),
        base_backend=resolved.base_backend,
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
    log_dir = root.output.working_dir.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_file_logging(
        working_dir=log_dir,
        prefix=root.output.out_file_prefix,
        level=root.output.log_level,
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

    from jaxrens.sampling.batch_descriptor import (
        ShardedSingleRun,
        SingleRun,
    )
    from jaxrens.cli.run import (
        run_multi_gpu_from_config,
        run_sharded_from_config,
    )

    resolved = resolve(root)

    if isinstance(resolved.batcher, SingleRun):
        pressure = resolved.ensemble_params_per_run[0].get("pressure")
        logger.info(
            "[single-run] seed=%d%s",
            resolved.ns.seed,
            f" pressure={pressure:.4g}" if pressure is not None else "",
        )
        _run_single(resolved)
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
        run_sharded_from_config(resolved)
    else:
        logger.info(
            "[multi-replica] n_gpu=%d n_per_gpu=%d n_total=%d pressures=%s",
            resolved.ns.n_gpu,
            resolved.ns.n_per_gpu,
            resolved.ns.n_gpu * resolved.ns.n_per_gpu,
            ", ".join(
                f"{p.get('pressure'):.4g}"
                if p.get("pressure") is not None else "—"
                for p in resolved.ensemble_params_per_run
            ),
        )
        run_multi_gpu_from_config(resolved)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    root = _load_and_validate(args.config, args.set)

    if args.parse_only:
        n_moves = len(root.moves)
        move_types = ", ".join(m.move_type for m in root.moves)
        print(
            f"OK — schema validation passed (parse-only)\n"
            f"  run:     n_live={root.run.n_live}, "
            f"max_iterations={root.run.max_iterations}\n"
            f"  moves:   {n_moves} move(s) [{move_types}]\n"
            f"  backend: {root.backend.backend_type}\n"
            f"  output:  format={root.output.format}, "
            f"prefix={root.output.out_file_prefix}"
        )
        return 0

    from jaxrens.sampling.batch_descriptor import (
        ShardedSingleRun,
        SingleRun,
    )

    resolved = resolve(root)
    n_moves = len(resolved.moves)
    move_types = ", ".join(m.move_type for m in resolved.moves)
    n_atoms = int(resolved.init.initial_positions.shape[-2])

    if isinstance(resolved.batcher, SingleRun):
        topology_line = "  topology: SingleRun (1 replica, 1 GPU)\n"
    elif isinstance(resolved.batcher, ShardedSingleRun):
        topology_line = (
            f"  topology: ShardedSingleRun "
            f"(1 replica, sharded across {resolved.batcher.n_gpu} GPUs)\n"
        )
    else:
        topology_line = (
            f"  topology: n_gpu={resolved.ns.n_gpu} × "
            f"n_per_gpu={resolved.ns.n_per_gpu} = "
            f"{resolved.ns.n_gpu * resolved.ns.n_per_gpu} replica(s)\n"
        )

    print(
        f"OK\n"
        f"{topology_line}"
        f"  run:     n_live={resolved.ns.n_live}, "
        f"max_iterations={resolved.ns.max_iterations}\n"
        f"  moves:   {n_moves} move(s) [{move_types}]\n"
        f"  backend: {resolved.backend.backend_type}, n_atoms={n_atoms}\n"
        f"  output:  format={resolved.output.format}, "
        f"prefix={resolved.output.out_file_prefix}"
    )
    return 0


def _cmd_dump_schema(args: argparse.Namespace) -> int:
    schema = RootSpec.model_json_schema()
    fmt = getattr(args, "format", "json")
    if fmt == "json":
        print(json.dumps(schema, indent=2))
    return 0


def _cmd_migrate_ns_inp(args: argparse.Namespace) -> int:
    """Migrate an old ns.inp file to jaxrens YAML format.

    Reads key=value lines from *args.input* (file path or stdin), converts
    them via ``migrate_ns_inp``, and writes YAML to *args.output* (file path
    or stdout).  All warnings/info go to stderr so that a shell redirect of
    stdout captures clean YAML.

    If ``--validate`` is passed, the migrated YAML is round-tripped through
    ``RootSpec.model_validate``; any validation error is printed to stderr
    and the command returns exit code 1.
    """
    from jaxrens.cli.migrate import migrate_ns_inp
    from jaxrens.cli.parser import parse_input_file
    import io
    import tempfile

    # Read input ----------------------------------------------------------
    if args.input == "-":
        raw_text = sys.stdin.read()
        # parse_input_file needs a file; write to a temp file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".inp", delete=False) as tmp:
            tmp.write(raw_text)
            tmp_path = tmp.name
        raw = parse_input_file(tmp_path)
        import os
        os.unlink(tmp_path)
    else:
        raw = parse_input_file(args.input)

    # Migrate -------------------------------------------------------------
    result = migrate_ns_inp(raw)
    cfg_dict: dict[str, Any] = result["config"]
    logs: list[dict[str, str]] = result["logs"]

    # Emit diagnostics to stderr ------------------------------------------
    for entry in logs:
        print(f"[{entry['level']}] {entry['message']}", file=sys.stderr)

    # Build YAML, hoisting _unknown keys as comments ----------------------
    unknown: dict[str, str] = cfg_dict.pop("_unknown", {})

    yaml_text = yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False)

    if unknown:
        comment_lines = ["# UNKNOWN keys from old ns.inp — review manually:"]
        for k, v in unknown.items():
            comment_lines.append(f"# UNKNOWN: {k}={v}")
        yaml_text = "\n".join(comment_lines) + "\n" + yaml_text

    # Write output --------------------------------------------------------
    if args.output == "-":
        sys.stdout.write(yaml_text)
    else:
        Path(args.output).write_text(yaml_text)

    # Optional validation round-trip --------------------------------------
    if args.validate:
        try:
            RootSpec.model_validate(yaml.safe_load(yaml_text))
        except Exception as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            return 1
        print("Validation OK", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxrens",
        description="jaxrens nested sampling toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Run nested sampling from a YAML config.")
    p_run.add_argument("-c", "--config", required=True, metavar="FILE",
                       help="Path to YAML config file.")
    p_run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="Override a config value (may be repeated; later wins).")
    p_run.add_argument(
        "--n-gpus", type=int, default=None, dest="n_gpus", metavar="N",
        help=(
            "Assert that JAX sees exactly N local GPU devices; exit non-zero "
            "with a diagnostic on mismatch.  Typically passed from SLURM as "
            "--n-gpus $SLURM_GPUS_ON_NODE so jobs fail fast when the scheduler "
            "downgrades the GPU allocation silently."
        ),
    )

    # -- validate --
    p_val = sub.add_parser("validate", help="Validate a YAML config without running.")
    p_val.add_argument("-c", "--config", required=True, metavar="FILE")
    p_val.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
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
    p_dump = sub.add_parser("dump-schema", help="Print the JSON schema for RootSpec.")
    p_dump.add_argument("--format", choices=["json"], default="json")

    # -- migrate-ns-inp --
    p_mig = sub.add_parser(
        "migrate-ns-inp",
        help="Convert an old ns.inp key=value file to jaxrens YAML format.",
    )
    p_mig.add_argument(
        "-i", "--input",
        default="-",
        metavar="INPUT.inp",
        help="Path to the old ns.inp file (default: stdin).",
    )
    p_mig.add_argument(
        "-o", "--output",
        default="-",
        metavar="OUTPUT.yaml",
        help="Path for the output YAML file (default: stdout).",
    )
    p_mig.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help=(
            "After migrating, round-trip through RootSpec.model_validate and "
            "exit non-zero if validation fails."
        ),
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
    model_cls: Any, prefix: str = "", _seen: set | None = None,
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

    all_paths = _collect_field_paths(RootSpec)

    exact = [p for p in all_paths if p.rsplit(".", 1)[-1] == bad_key]
    parent_fields = [
        p.rsplit(".", 1)[-1] if "." in p else p
        for p in all_paths
        if p.rsplit(".", 1)[0] == parent_path or (parent_path == "" and "." not in p)
    ]
    fuzzy = difflib.get_close_matches(bad_key, parent_fields, n=3, cutoff=0.6)
    fuzzy_paths = [
        (f"{parent_path}.{f}" if parent_path else f) for f in fuzzy if f != bad_key
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
        "migrate-ns-inp": _cmd_migrate_ns_inp,
    }
    handler = dispatch[args.command]
    try:
        sys.exit(handler(args))
    except ValidationError as exc:
        cfg = getattr(args, "config", "<unknown>")
        print(_format_validation_error(exc, cfg), file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as exc:
        print(f"jaxrens: {exc}", file=sys.stderr)
        sys.exit(2)
    except yaml.YAMLError as exc:
        print(f"jaxrens: YAML parse error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
