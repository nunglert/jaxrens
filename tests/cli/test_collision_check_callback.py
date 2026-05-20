"""Tests for ``CollisionCheckCallback`` (atom-pair distance warning).

Covers:
- PBC walkers with all atoms well-separated: no warning.
- PBC walkers with one colliding pair: warning fires, worst-offender
  (run, walker) attribution is correct.
- Open-boundary (cell is None) variant detects collisions.
- ``interval`` gates firing: only iterations that are multiples emit
  warnings, and the JIT path is not even invoked otherwise.
- VmapRuns shape conventions: multi-run attribution remains correct.
"""

from __future__ import annotations

import logging
import types

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.cli.monitor import CollisionCheckCallback
from jaxrens.sampling.batch_descriptor import SingleRun, VmapRuns


def _capture_warnings(cb_action):
    records: list[str] = []

    class _Rec(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Rec()
    target = logging.getLogger("jaxrens.cli.monitor")
    target.addHandler(h)
    old = target.level
    target.setLevel(logging.WARNING)
    try:
        cb_action()
    finally:
        target.removeHandler(h)
        target.setLevel(old)
    return records


def _fake_ns_state(positions, cell):
    pop = types.SimpleNamespace(positions=positions, cell=cell)
    return types.SimpleNamespace(population=pop)


# ---------------------------------------------------------------------------
# SingleRun shape (K, N, 3)
# ---------------------------------------------------------------------------


class TestSingleRunPBC:
    def test_clean_lattice_no_warn(self):
        # 4 atoms on a simple cubic grid of spacing 1.0 in a 4-Å cell;
        # all pair distances ≥ 1.0 (and PBC images at distance 3.0 +).
        positions = jnp.array(
            [[[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]],
            dtype=jnp.float32,
        )  # (K=1, N=4, 3)
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 4.0  # (1, 3, 3)
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(lambda: cb.on_iteration(
            iteration=1, ns_state=ns_state, info={"_batcher": SingleRun()},
        ))
        assert msgs == []

    def test_collision_warns_with_attribution(self):
        # K=2 walkers; walker 1 has atoms 1.0 Å apart (clean); walker 0 has
        # atoms 0.05 Å apart (collision).
        clean = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        collided = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 1.0, 0.0]]
        positions = jnp.array([collided, clean], dtype=jnp.float32)
        cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 4.0, (2, 3, 3))
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(lambda: cb.on_iteration(
            iteration=1, ns_state=ns_state, info={"_batcher": SingleRun()},
        ))
        assert len(msgs) == 1
        # Worst offender lives in run 0 (SingleRun gets a length-1 run axis),
        # walker 0.
        assert "run[0].walker[0]" in msgs[0]
        assert "d_min=0.05" in msgs[0]
        assert "below 0.5" in msgs[0]


class TestSingleRunOpen:
    def test_open_boundary_detects_collision(self):
        # cell=None branch.  Two atoms 0.1 Å apart.
        positions = jnp.array(
            [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0]]],
            dtype=jnp.float32,
        )
        ns_state = _fake_ns_state(positions, cell=None)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(lambda: cb.on_iteration(
            iteration=1, ns_state=ns_state, info={"_batcher": SingleRun()},
        ))
        assert len(msgs) == 1
        assert "run[0].walker[0]" in msgs[0]
        assert "d_min=0.1" in msgs[0]

    def test_open_boundary_clean_no_warn(self):
        positions = jnp.array(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=jnp.float32,
        )
        ns_state = _fake_ns_state(positions, cell=None)
        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(lambda: cb.on_iteration(
            iteration=1, ns_state=ns_state, info={"_batcher": SingleRun()},
        ))
        assert msgs == []


# ---------------------------------------------------------------------------
# Interval gating
# ---------------------------------------------------------------------------


class TestIntervalGating:
    def test_only_multiples_of_interval_fire(self):
        # Always-colliding configuration; we should only see warnings on
        # iterations 0, 10, 20 with interval=10.
        positions = jnp.array(
            [[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]],
            dtype=jnp.float32,
        )
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 4.0
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=10)
        msgs = _capture_warnings(lambda: [
            cb.on_iteration(i, ns_state, {"_batcher": SingleRun()})
            for i in range(25)
        ])
        # 0, 10, 20 → 3 fires.
        assert len(msgs) == 3


# ---------------------------------------------------------------------------
# Multi-run (VmapRuns) shape (R, K, N, 3)
# ---------------------------------------------------------------------------


class TestVmapRunsShape:
    def test_multi_run_attribution(self):
        # 2 runs × 2 walkers each.  Collision sits at run=1, walker=0.
        clean = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        collided = [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 1.0, 0.0]]
        positions = jnp.array(
            [[clean, clean], [collided, clean]],
            dtype=jnp.float32,
        )  # (R=2, K=2, N=3, 3)
        cell = jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float32) * 4.0, (2, 2, 3, 3),
        )
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(lambda: cb.on_iteration(
            iteration=1, ns_state=ns_state, info={"_batcher": VmapRuns(2)},
        ))
        assert len(msgs) == 1
        assert "run[1].walker[0]" in msgs[0]
        assert "d_min=0.03" in msgs[0]


# ---------------------------------------------------------------------------
# JIT exercise (the inner computation must be traceable)
# ---------------------------------------------------------------------------


class TestJITPath:
    def test_jit_compiles_pbc_path(self):
        positions = jnp.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        )
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 4.0
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        # Two firings → second must reuse the cached trace; no exception.
        cb.on_iteration(0, ns_state, {"_batcher": SingleRun()})
        cb.on_iteration(1, ns_state, {"_batcher": SingleRun()})

    def test_jit_compiles_open_path(self):
        positions = jnp.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        )
        ns_state = _fake_ns_state(positions, cell=None)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        cb.on_iteration(0, ns_state, {"_batcher": SingleRun()})
        cb.on_iteration(1, ns_state, {"_batcher": SingleRun()})
