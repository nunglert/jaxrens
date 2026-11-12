"""Tests for ``CollisionCheckCallback`` in ``jaxrens.cli.monitor``.

The callback logs a warning when a walker has two atoms closer than the
threshold, naming the worst (run, walker) pairs. The tests use a fake NS
state with periodic and open cells, the ``interval`` gate, and the
multi-run layout of ``VmapRuns``.
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


class TestSingleRunPBC:
    """Check one run in a periodic cell."""

    def test_clean_lattice_no_warn(self):
        # 4 atoms on a simple cubic grid of spacing 1.0 in a 4-Å cell;
        # all pair distances ≥ 1.0 (and PBC images at distance 3.0 +).
        positions = jnp.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=jnp.float32,
        )  # (K=1, N=4, 3)
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 4.0  # (1, 3, 3)
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(
            lambda: cb.on_iteration(
                iteration=1,
                ns_state=ns_state,
                info={"_batcher": SingleRun()},
            )
        )
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
        msgs = _capture_warnings(
            lambda: cb.on_iteration(
                iteration=1,
                ns_state=ns_state,
                info={"_batcher": SingleRun()},
            )
        )
        assert len(msgs) == 1
        # Worst offender lives in run 0 (SingleRun gets a length-1 run axis),
        # walker 0.
        assert "run[0].walker[0]" in msgs[0]
        assert "d_min=0.05" in msgs[0]
        assert "below 0.5" in msgs[0]

    def test_collision_across_cell_boundary_warns(self):
        # 0.15 Å apart through the periodic boundary, 3.85 Å apart inside
        # the 4 Å cell.
        positions = jnp.array(
            [[[0.1, 0.0, 0.0], [3.95, 0.0, 0.0]]], dtype=jnp.float32
        )
        cell = jnp.eye(3, dtype=jnp.float32)[None] * 4.0
        ns_state = _fake_ns_state(positions, cell)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(
            lambda: cb.on_iteration(
                iteration=1,
                ns_state=ns_state,
                info={"_batcher": SingleRun()},
            )
        )
        assert len(msgs) == 1
        assert "d_min=0.15" in msgs[0]


class TestSingleRunOpen:
    """Check one run without a cell (open boundaries)."""

    def test_open_boundary_detects_collision(self):
        # Two atoms 0.1 Å apart.
        positions = jnp.array(
            [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 5.0]]],
            dtype=jnp.float32,
        )
        ns_state = _fake_ns_state(positions, cell=None)

        cb = CollisionCheckCallback(threshold=0.5, interval=1)
        msgs = _capture_warnings(
            lambda: cb.on_iteration(
                iteration=1,
                ns_state=ns_state,
                info={"_batcher": SingleRun()},
            )
        )
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
        msgs = _capture_warnings(
            lambda: cb.on_iteration(
                iteration=1,
                ns_state=ns_state,
                info={"_batcher": SingleRun()},
            )
        )
        assert msgs == []


def test_only_multiples_of_interval_fire():
    # Always colliding, so with interval=10 the warnings come at
    # iterations 0, 10 and 20.
    positions = jnp.array(
        [[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    cell = jnp.eye(3, dtype=jnp.float32)[None] * 4.0
    ns_state = _fake_ns_state(positions, cell)

    cb = CollisionCheckCallback(threshold=0.5, interval=10)
    msgs = _capture_warnings(
        lambda: [
            cb.on_iteration(i, ns_state, {"_batcher": SingleRun()})
            for i in range(25)
        ]
    )
    assert len(msgs) == 3


def test_multi_run_attribution():
    # 2 runs × 2 walkers each.  Collision sits at run=1, walker=0.
    clean = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    collided = [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 1.0, 0.0]]
    positions = jnp.array(
        [[clean, clean], [collided, clean]],
        dtype=jnp.float32,
    )  # (R=2, K=2, N=3, 3)
    cell = jnp.broadcast_to(
        jnp.eye(3, dtype=jnp.float32) * 4.0,
        (2, 2, 3, 3),
    )
    ns_state = _fake_ns_state(positions, cell)

    cb = CollisionCheckCallback(threshold=0.5, interval=1)
    msgs = _capture_warnings(
        lambda: cb.on_iteration(
            iteration=1,
            ns_state=ns_state,
            info={"_batcher": VmapRuns(2)},
        )
    )
    assert len(msgs) == 1
    assert "run[1].walker[0]" in msgs[0]
    assert "d_min=0.03" in msgs[0]
