"""Single-backend ns_step benchmark.

Reads one bench YAML, resolves to runtime objects, jits ``ns_step``, then
records (a) a cold-call wall time that includes the JIT compile and
(b) per-step warm steady-state wall times.  One row appended to
``bench/results.jsonl``.

Invoked as a subprocess by ``bench/run.py`` — one process per backend
so each one's cold-compile time is uncontaminated by previous backends'
JAX startup amortisation.

Exit codes:
    0   success — row appended.
    77  skipped — optional dep / model fixture missing for this backend.
    1   error — uncaught exception during setup / timing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("bench._one")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "bench" / "results.jsonl"


#: Env var carrying an authoritative commit SHA, overriding the git probe.
#: CI sets it because the bench runs inside a container over a bind-mounted
#: checkout: the container is root, the mount is owned by the runner's uid, and
#: git then refuses the repo with "detected dubious ownership", so the probe
#: below returns None and the comparison renders "Comparing PR ? against base ?".
#: The workflow already knows both SHAs, so it passes them in rather than having
#: us re-derive something it can state authoritatively.
_SHA_ENV = "JAXRENS_BENCH_SHA"

_GIT_WARNED = False


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo, returning stdout, or None on failure.

    Failures are logged (once per process) rather than swallowed: this probe
    silently returning None is what let the "?" SHAs reach a rendered PR
    comment unnoticed.
    """
    global _GIT_WARNED
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        if not _GIT_WARNED:
            _GIT_WARNED = True
            stderr = getattr(exc, "stderr", "") or ""
            logger.warning(
                "bench: `git %s` failed, provenance fields will be null: %s%s",
                " ".join(args),
                exc,
                f" -- {stderr.strip()}" if stderr.strip() else "",
            )
        return None


def _git_sha() -> str | None:
    """Short commit SHA: the ``JAXRENS_BENCH_SHA`` override, else a git probe.

    Normalised to 7 characters either way — ``compare.py`` slices to 7 and
    ``view.py`` formats the column at width 8, and the archived rows in
    ``results.jsonl`` have always held the short form.
    """
    override = os.environ.get(_SHA_ENV, "").strip()
    if override:
        return override[:7]
    sha = _run_git("rev-parse", "--short", "HEAD")
    return sha[:7] if sha else None


def _git_dirty() -> bool | None:
    """Whether the worktree has uncommitted changes (None if git is unusable).

    Deliberately *not* forced to False when ``JAXRENS_BENCH_SHA`` is set: CI
    deletes ``bench/results.jsonl`` between runs, so the tree genuinely is
    dirty relative to that SHA, and None is the honest answer when we cannot
    ask git.
    """
    out = _run_git("status", "--porcelain")
    return bool(out) if out is not None else None


def _env_fingerprint() -> dict:
    import jax
    import jaxlib

    try:
        import jaxrens

        jaxrens_version = getattr(jaxrens, "__version__", "unknown")
    except ImportError:
        jaxrens_version = "unknown"

    devices = jax.local_devices()
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jaxrens_version": jaxrens_version,
        "n_devices": len(devices),
        "device_kind": str(devices[0].device_kind) if devices else "none",
        "platform": str(jax.default_backend()),
        "hostname": socket.gethostname(),
    }


def _run_bench(
    backend_name: str, config_path: Path, n_timed_steps: int
) -> dict:
    """Build → cold call → warm steady-state.  Return the row dict."""
    # Imports deferred to inside the function so a missing optional dep
    # (e.g. mace_jax) raises ImportError here, where the caller catches
    # it and exits 77 — not at module top.
    import jax
    import jax.numpy as jnp

    from jaxrens.backends.ensemble import EnsembleBackend
    from jaxrens.cli.resolve import resolve
    from jaxrens.cli.run import _to_runtime_ensemble_params
    from jaxrens.cli.schema import RootSpec
    from jaxrens.sampling.batch_descriptor import SingleRun
    from jaxrens.sampling.mwg import build_mwg
    from jaxrens.sampling.nested_sampling import (
        _choose_starting_bucket,
        init_ns,
        ns_step,
    )

    # ---- Setup phase --------------------------------------------------------
    t_setup_start = time.perf_counter()

    root = RootSpec.model_validate(yaml.safe_load(config_path.read_text()))
    resolved = resolve(root)

    # Wrap with EnsembleBackend exactly as cli/run.py:run_from_config does:
    # take the resolved per-run ensemble params (NVT, NPT, or semi-grand μPT)
    # and wrap the base backend once when any are present.
    base_backend = resolved.base_backend
    ensemble_params = _to_runtime_ensemble_params(
        resolved.ensemble_params_per_run[0]
        if resolved.ensemble_params_per_run
        else None
    )
    if ensemble_params is not None:
        step_backend = EnsembleBackend(base_backend, pressure=0.0)
    else:
        step_backend = base_backend

    init_fn, step_fn, _per_move_fns = build_mwg(
        step_backend, list(resolved.move_descriptors)
    )

    init = resolved.init
    ladder = tuple(int(x) for x in resolved.backend.max_neighbors_list)
    offset = int(resolved.backend.max_neighbors_offset)
    starting_bucket = _choose_starting_bucket(
        init.initial_max_neighbor_counts,
        ladder,
        offset,
    )

    key = jax.random.key(resolved.ns.seed)
    key_init, key_steps = jax.random.split(key, 2)

    ns_state = init_ns(
        init_fn,
        init.initial_positions,
        init.initial_types,
        init.initial_energies,
        init.initial_cells,
        key_init,
        ensemble_params=ensemble_params,
        max_neighbors=starting_bucket,
        max_neighbor_counts=init.initial_max_neighbor_counts,
    )

    # Production wraps via batch_descriptor; for a single-run bench
    # SingleRun.wrap_step is `jax.jit(ns_step, static_argnums=(1,2,3))`
    # — i.e. exactly the call pattern `run_loop._run_loop` uses.
    batcher = SingleRun()
    jit_step = batcher.wrap_step(
        ns_step,
        step_fn,
        resolved.ns.n_mcmc_steps,
        resolved.ns.n_extra,
    )

    t_setup = time.perf_counter() - t_setup_start

    # ---- Cold-call phase ----------------------------------------------------
    n_mcmc = resolved.ns.n_mcmc_steps
    n_extra = resolved.ns.n_extra

    t0 = time.perf_counter()
    ns_state, _info = jit_step(ns_state, step_fn, n_mcmc, n_extra)
    jax.block_until_ready(ns_state.population.energy)
    t_cold = time.perf_counter() - t0

    # ---- Warm steady-state phase --------------------------------------------
    per_step: list[float] = []
    for _ in range(n_timed_steps):
        t0 = time.perf_counter()
        ns_state, _info = jit_step(ns_state, step_fn, n_mcmc, n_extra)
        jax.block_until_ready(ns_state.population.energy)
        per_step.append(time.perf_counter() - t0)

    per_step_arr = np.asarray(per_step, dtype=np.float64)

    # ---- Build row ----------------------------------------------------------
    n_atoms = int(init.initial_positions.shape[-2])
    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "backend": backend_name,
        "config_file": str(config_path.relative_to(REPO_ROOT)),
        "scale": {
            "n_live": int(resolved.ns.n_live),
            "n_atoms": n_atoms,
            "n_mcmc_steps": int(n_mcmc),
            "n_extra": int(n_extra),
            "n_moves": len(resolved.move_descriptors),
            "starting_bucket": int(starting_bucket),
            "pressure_eva3": (
                float(ensemble_params["pressure"])
                if ensemble_params and "pressure" in ensemble_params
                else None
            ),
        },
        "n_warmup_steps": 1,
        "n_timed_steps": n_timed_steps,
        "t_setup_s": float(t_setup),
        "t_cold_compile_s": float(t_cold),
        "t_warm_per_step_s": {
            "mean": float(per_step_arr.mean()),
            "std": float(per_step_arr.std(ddof=0)),
            "p50": float(np.percentile(per_step_arr, 50)),
            "p95": float(np.percentile(per_step_arr, 95)),
            "min": float(per_step_arr.min()),
        },
        "env": _env_fingerprint(),
    }
    return row


def _append_row(row: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        help="Backend label (used for the result row).",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the bench YAML for this backend.",
    )
    parser.add_argument(
        "--n-timed-steps",
        type=int,
        default=50,
        help="Steady-state iterations after the cold call.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"[bench] config not found: {args.config}", file=sys.stderr)
        return 1

    try:
        row = _run_bench(
            args.backend, args.config.resolve(), args.n_timed_steps
        )
    except ImportError as e:
        print(
            f"[bench] skip {args.backend}: missing dep — {e}", file=sys.stderr
        )
        return 77
    except FileNotFoundError as e:
        # Model-fixture path (neuralil .pkl, nequix model dir).  Surfaces here
        # when the resolver instantiates the backend.
        print(
            f"[bench] skip {args.backend}: missing fixture — {e}",
            file=sys.stderr,
        )
        return 77

    _append_row(row)

    # Brief one-line summary on stdout.
    warm = row["t_warm_per_step_s"]
    print(
        f"[bench] {args.backend}: setup={row['t_setup_s']:.2f}s  "
        f"cold={row['t_cold_compile_s']:.2f}s  "
        f"warm.mean={warm['mean']*1e3:.2f}ms  "
        f"warm.p50={warm['p50']*1e3:.2f}ms  "
        f"warm.p95={warm['p95']*1e3:.2f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
