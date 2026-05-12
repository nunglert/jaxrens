"""Diff two bench/results.jsonl files and print a markdown comparison.

Stdlib only — the CI workflow runs this outside the docker image, so
adding deps here would mean adding them to the runner.

Usage:
    python bench/compare.py <base.jsonl> <pr.jsonl> [--metric warm|cold]

Output: a markdown block on stdout, suitable for posting as a PR
comment.  The first line is an HTML comment marker
``<!-- jaxrens-bench -->`` so the find-comment step in the GHA workflow
can locate prior comments unambiguously.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Order rows appear in the comment, even if missing from both files.
BACKEND_ORDER = ("lj", "jaxmd", "mace", "neuralil", "nequix")


def _load_latest_per_backend(path: Path) -> dict[str, dict]:
    """Read JSONL; for each backend keep the last row by timestamp."""
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            b = row.get("backend")
            if not b:
                continue
            prev = rows.get(b)
            if prev is None or row.get("timestamp_utc", "") >= prev.get(
                "timestamp_utc", ""
            ):
                rows[b] = row
    return rows


def _fmt_ms(s: float | None) -> str:
    return f"{s * 1e3:.2f}" if s is not None else "—"


def _fmt_s(s: float | None) -> str:
    return f"{s:.2f}" if s is not None else "—"


def _fmt_delta_pct(base: float | None, new: float | None) -> str:
    if base is None or new is None or base == 0.0:
        return "—"
    pct = 100.0 * (new - base) / base
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f} %"


def _env_summary(row: dict | None) -> str:
    if row is None:
        return "(no rows)"
    env = row.get("env", {})
    dev = env.get("device_kind", "?")
    nd = env.get("n_devices", "?")
    jv = env.get("jax_version", "?")
    return f"{dev} ({nd} device(s)), JAX {jv}"


def _short_sha(row: dict | None) -> str:
    if row is None:
        return "?"
    return (row.get("git_sha") or "?")[:7]


def render(base: dict[str, dict], pr: dict[str, dict], base_failed: bool = False) -> str:
    # Pick an env line from whichever side has rows.  Prefer PR.
    any_pr = next(iter(pr.values()), None)
    any_base = next(iter(base.values()), None)
    env_line = _env_summary(any_pr or any_base)
    base_sha = _short_sha(any_base)
    pr_sha = _short_sha(any_pr)

    lines: list[str] = []
    lines.append("<!-- jaxrens-bench -->")
    lines.append("## Benchmark result — `ns_step`")
    lines.append("")
    if base_failed:
        lines.append(
            "> ⚠️  **Base bench failed or skipped.**  The base SHA either "
            "doesn't carry `bench/` yet (typical for the PR that first "
            "introduces this workflow) or its bench run errored.  Showing "
            "PR-only numbers — Δ columns are not meaningful here."
        )
        lines.append("")
    lines.append(
        f"Comparing PR `{pr_sha}` against base `{base_sha}`.  "
        f"Hardware: {env_line}."
    )
    lines.append("")
    lines.append("| backend | n_live | base p50 [ms] | PR p50 [ms] | Δ p50 |")
    lines.append("|---------|-------:|--------------:|------------:|------:|")

    cold_bits: list[str] = []
    for b in BACKEND_ORDER:
        br = base.get(b)
        pr_r = pr.get(b)

        if br is None and pr_r is None:
            lines.append(f"| {b} | — | — | — | _skipped both_ |")
            continue

        n_live = "—"
        for r in (pr_r, br):
            if r is not None:
                n_live = str(r.get("scale", {}).get("n_live", "—"))
                break

        base_p50 = br["t_warm_per_step_s"]["p50"] if br is not None else None
        pr_p50 = pr_r["t_warm_per_step_s"]["p50"] if pr_r is not None else None

        if br is None:
            delta = "_added_"
        elif pr_r is None:
            delta = "_removed_"
        else:
            delta = _fmt_delta_pct(base_p50, pr_p50)

        lines.append(
            f"| {b} | {n_live} | {_fmt_ms(base_p50)} | {_fmt_ms(pr_p50)} | {delta} |"
        )

        if br is not None and pr_r is not None:
            cold_bits.append(
                f"{b} {_fmt_s(br['t_cold_compile_s'])}→"
                f"{_fmt_s(pr_r['t_cold_compile_s'])} s"
            )

    lines.append("")
    if cold_bits:
        lines.append("Cold-compile (one-shot, noisier): " + ", ".join(cold_bits) + ".")
        lines.append("")
    lines.append(
        "> CI bench runs both branches on the same ephemeral g5.xlarge spot "
        "instance.  Treat |Δ| below ~10 % as noise; |Δ| above ~25 % is a "
        "likely real change worth investigating before merge.  Comment is "
        "informational; CI does not fail on Δ."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Baseline results.jsonl")
    parser.add_argument("pr", type=Path, help="PR-head results.jsonl")
    parser.add_argument(
        "--base-failed",
        action="store_true",
        help="Mark the base run as failed/skipped — emits a warning banner "
        "and disclaims the Δ columns.",
    )
    args = parser.parse_args(argv)

    base = _load_latest_per_backend(args.base)
    pr = _load_latest_per_backend(args.pr)

    if not base and not pr and not args.base_failed:
        print(
            "[bench-compare] both inputs empty — nothing to compare",
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(render(base, pr, base_failed=args.base_failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
