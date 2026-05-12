"""Orchestrator: run ns_step bench for each backend (one subprocess each).

Sequential — one GPU process at a time (per CLAUDE.md).  Subprocess
isolation means each backend's cold-compile timing carries the same
fixed JAX startup cost, and a missing optional dep can't poison the
other backends.

Usage:
    python bench/run.py                   # all backends with a config in
                                          # bench/configs/.
    python bench/run.py --backend jaxmd   # one backend.
    python bench/run.py --n-timed-steps 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONE = HERE / "_one.py"
CONFIGS = HERE / "configs"

# Backends in run order.  Cheap first so a partial sweep still produces
# the most data points.
KNOWN_BACKENDS: tuple[str, ...] = ("lj", "jaxmd", "mace", "neuralil", "nequix")

EXIT_SKIPPED = 77


def _resolve_backends(selected: list[str] | None) -> list[str]:
    if selected:
        for b in selected:
            if b not in KNOWN_BACKENDS:
                raise SystemExit(
                    f"[bench] unknown backend {b!r}; known: {KNOWN_BACKENDS}"
                )
        return list(selected)
    return list(KNOWN_BACKENDS)


def _run_one(backend: str, n_timed_steps: int) -> int:
    """Spawn _one.py for one backend.  Return its exit code."""
    cfg = CONFIGS / f"{backend}.yaml"
    if not cfg.exists():
        print(f"[bench] {backend}: no config at {cfg.relative_to(HERE.parent)}; "
              f"skipping")
        return EXIT_SKIPPED

    print(f"\n[bench] === {backend} === (config: {cfg.name})", flush=True)
    proc = subprocess.run(
        [sys.executable, str(ONE),
         "--backend", backend,
         "--config", str(cfg),
         "--n-timed-steps", str(n_timed_steps)],
        # Inherit stdout/stderr so progress is visible live; _one.py
        # prints the summary line itself.
    )
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", default=None,
                        help="Run only this backend.  Repeatable.")
    parser.add_argument("--n-timed-steps", type=int, default=50,
                        help="Forwarded to _one.py.")
    args = parser.parse_args(argv)

    backends = _resolve_backends(args.backend)

    ok = 0
    skipped = 0
    failed: list[str] = []

    for b in backends:
        rc = _run_one(b, args.n_timed_steps)
        if rc == 0:
            ok += 1
        elif rc == EXIT_SKIPPED:
            skipped += 1
        else:
            failed.append(b)

    print("\n[bench] " + " ".join([
        f"ok={ok}", f"skipped={skipped}", f"failed={len(failed)}",
    ]), flush=True)
    if failed:
        print(f"[bench] failed backends: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
