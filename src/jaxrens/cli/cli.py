"""argparse entry point for jaxrens.

Subcommands
-----------
run        Load YAML, validate, resolve, execute NS.
validate   Load YAML, validate only; print OK summary.
dump-schema  Print JSON schema for RootConfig.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from jaxrens.cli.resolve import expand_cohort, resolve
from jaxrens.cli.run import run_from_config
from jaxrens.cli.schema import RootConfig


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

def _load_and_validate(config_path: str, overrides: list[str]) -> RootConfig:
    with open(config_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    if overrides:
        raw = _apply_overrides(raw, overrides)
    return RootConfig.model_validate(raw)


def _run_one(resolved, *, cohort_label: str = "") -> None:
    """Execute a single NS run from a ``ResolvedConfig``."""
    run_from_config(
        resolved.ns,
        list(resolved.moves),
        resolved.backend,
        resolved.output,
        initial_positions=resolved.init.initial_positions,
        initial_types=resolved.init.initial_types,
        initial_energies=resolved.init.initial_energies,
        initial_cells=resolved.init.initial_cells,
        restart_state=resolved.init.restart_state,
        move_descriptors=list(resolved.move_descriptors),
        initial_walk_config=resolved.initial_walk_config,
        adaptation_config=resolved.adaptation_cfg,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    root = _load_and_validate(args.config, args.set)
    cohort = expand_cohort(root)
    n = len(cohort)

    if n == 1:
        _run_one(cohort[0])
    else:
        for i, resolved in enumerate(cohort):
            print(f"[cohort {i + 1}/{n}] pressure={resolved.ensemble_params.get('pressure')} seed={resolved.ns.seed}")
            _run_one(resolved, cohort_label=f"{i + 1}/{n}")

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    root = _load_and_validate(args.config, args.set)
    cohort = expand_cohort(root)
    n = len(cohort)
    resolved = cohort[0]
    n_moves = len(resolved.moves)
    move_types = ", ".join(m.move_type for m in resolved.moves)
    n_atoms = int(resolved.init.initial_positions.shape[-2])
    print(
        f"OK — cohort size: {n}\n"
        f"  run:     n_live={resolved.ns.n_live}, "
        f"max_iterations={resolved.ns.max_iterations}\n"
        f"  moves:   {n_moves} move(s) [{move_types}]\n"
        f"  backend: {resolved.backend.backend_type}, n_atoms={n_atoms}\n"
        f"  output:  format={resolved.output.format}, "
        f"prefix={resolved.output.out_file_prefix}"
    )
    return 0


def _cmd_dump_schema(args: argparse.Namespace) -> int:
    schema = RootConfig.model_json_schema()
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
    ``RootConfig.model_validate``; any validation error is printed to stderr
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
            RootConfig.model_validate(yaml.safe_load(yaml_text))
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

    # -- validate --
    p_val = sub.add_parser("validate", help="Validate a YAML config without running.")
    p_val.add_argument("-c", "--config", required=True, metavar="FILE")
    p_val.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    # -- dump-schema --
    p_dump = sub.add_parser("dump-schema", help="Print the JSON schema for RootConfig.")
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
            "After migrating, round-trip through RootConfig.model_validate and "
            "exit non-zero if validation fails."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point registered via ``[project.scripts]``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "dump-schema": _cmd_dump_schema,
        "migrate-ns-inp": _cmd_migrate_ns_inp,
    }
    handler = dispatch[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
