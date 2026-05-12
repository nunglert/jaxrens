"""View ns_step benchmark results.

Reads ``bench/results.jsonl`` (one JSON object per line) and prints a
table or plots a trend over time.

Usage:
    python bench/view.py                       # whole log, table
    python bench/view.py --backend jaxmd       # one backend only
    python bench/view.py --last 20             # last 20 rows
    python bench/view.py --plot                # matplotlib trend chart
    python bench/view.py --plot --metric cold  # plot t_cold_compile_s instead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "bench" / "results.jsonl"


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _filter(rows: list[dict], backend: str | None, last: int | None) -> list[dict]:
    if backend is not None:
        rows = [r for r in rows if r.get("backend") == backend]
    if last is not None:
        rows = rows[-last:]
    return rows


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("[view] no rows")
        return

    hdr = (
        f"{'timestamp':<20}  {'sha':<8}  {'backend':<10}  "
        f"{'n_live':>6}  {'t_cold[s]':>9}  "
        f"{'t_warm.mean[ms]':>15}  {'t_warm.p95[ms]':>14}  hw"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        warm = r["t_warm_per_step_s"]
        env = r["env"]
        dirty_flag = "*" if r.get("git_dirty") else " "
        sha = (r.get("git_sha") or "—") + dirty_flag
        hw = f"{env.get('device_kind', '?')} (x{env.get('n_devices', '?')})"
        print(
            f"{r['timestamp_utc']:<20}  {sha:<8}  {r['backend']:<10}  "
            f"{r['scale']['n_live']:>6}  {r['t_cold_compile_s']:>9.3f}  "
            f"{warm['mean']*1e3:>15.3f}  {warm['p95']*1e3:>14.3f}  {hw}"
        )


def _plot_trend(rows: list[dict], metric: str) -> None:
    if not rows:
        print("[view] no rows to plot")
        return

    import matplotlib.pyplot as plt
    from datetime import datetime

    if metric == "warm":
        def value(r: dict) -> float:
            return r["t_warm_per_step_s"]["mean"] * 1e3
        ylabel = "t_warm_per_step_s.mean  [ms]"
    elif metric == "cold":
        def value(r: dict) -> float:
            return r["t_cold_compile_s"]
        ylabel = "t_cold_compile_s  [s]"
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    by_backend: dict[str, list[tuple[datetime, float]]] = {}
    for r in rows:
        ts = datetime.fromisoformat(r["timestamp_utc"].rstrip("Z"))
        by_backend.setdefault(r["backend"], []).append((ts, value(r)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for backend, pts in sorted(by_backend.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", label=backend, alpha=0.85)
    ax.set_xlabel("timestamp (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"ns_step benchmark trend — {metric}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None,
                        help="Filter rows to this backend.")
    parser.add_argument("--last", type=int, default=None,
                        help="Show only the last N rows after filtering.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the trend instead of printing a table.")
    parser.add_argument("--metric", choices=("warm", "cold"), default="warm",
                        help="Which metric to plot (--plot only).")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH,
                        help="Path to results.jsonl.")
    args = parser.parse_args(argv)

    rows = _filter(_load_rows(args.results), args.backend, args.last)

    if args.plot:
        _plot_trend(rows, args.metric)
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
