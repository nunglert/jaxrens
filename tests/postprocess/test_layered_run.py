"""Layered observables on every point of a run."""

import json
import logging

import ase
import ase.io
import h5py
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from jaxrens.cli.monitor import TrajectoryCallback
from jaxrens.io.energy_log import EnergyLogger
from jaxrens.io.trajectory import ExtxyzTrajectoryWriter
from jaxrens.postprocess import Monitor, layered_run
from jaxrens.postprocess.layered_run import (
    LayeredParams,
    configuration_observables,
    dead_point_rows,
    fill_dead_points,
    layered_observables,
    layered_params,
    load_layered_observables,
)
from jaxrens.state.walker import WalkerState
from tests._layered import LI, C, graphite

SYMBOLS = {C: "C", LI: "Li"}
PARAMS = LayeredParams(("C",), ("Li",), (3, 3))


class TestDeadPointRows:
    def test_single_cull(self):
        rows = dead_point_rows([0, 3, 6], None, n_cull=1)
        np.testing.assert_array_equal(rows, [0, 3, 6])

    def test_cull_ranks(self):
        rows = dead_point_rows([0, 0, 3, 3], [0, 1, 0, 1], n_cull=2)
        np.testing.assert_array_equal(rows, [0, 1, 6, 7])

    def test_matches_what_the_run_writes(self, tmp_path):
        """Frames written by the NS trajectory callback (n_cull = 2, a frame
        every 3 iterations) land on the .energies rows of the same dead
        points."""
        n_cull, n_atoms = 2, 3
        writer = ExtxyzTrajectoryWriter(tmp_path / "t.extxyz", {0: "Ar"})
        elog = EnergyLogger(tmp_path / "t.energies", n_walkers=8, n_cull=2)
        elog.write_header()
        callback = TrajectoryCallback(
            writer, elog, traj_interval=3, snapshot_interval=0
        )
        rng = np.random.default_rng(0)
        for it in range(8):
            energy = np.sort(rng.uniform(0, 1, n_cull))[::-1]
            dead = WalkerState(
                positions=jnp.asarray(rng.uniform(0, 5, (n_cull, n_atoms, 3))),
                types=jnp.zeros((n_cull, n_atoms), dtype=jnp.int32),
                energy=jnp.asarray(energy),
                cell=jnp.asarray(np.tile(np.eye(3) * 6.0, (n_cull, 1, 1))),
                n_atoms=n_atoms,
            )
            callback.on_iteration(it, None, {"dead_walkers": dead})
        callback.on_finish(None)

        frames = ase.io.read(tmp_path / "t.extxyz", index=":")
        log = EnergyLogger.read(tmp_path / "t.energies")
        rows = dead_point_rows(
            [a.info["iter"] for a in frames],
            [a.info["cull_rank"] for a in frames],
            log.n_cull,
        )
        np.testing.assert_array_equal(rows, [0, 1, 6, 7, 12, 13])
        np.testing.assert_allclose(
            [a.info["ns_energy"] for a in frames], log.energies[rows]
        )


class TestFillDeadPoints:
    def test_hold(self):
        out = fill_dead_points([10.0, 20.0, 30.0], [0, 3, 6], n_dead=8)
        np.testing.assert_array_equal(out, [10, 10, 10, 20, 20, 20, 30, 30])

    def test_hold_vectors_and_unsorted_frames(self):
        vals = np.array([[3.0, 3.5], [1.0, 1.5]])
        out = fill_dead_points(vals, [2, 0], n_dead=4)
        np.testing.assert_array_equal(out[:, 0], [1, 1, 3, 3])
        assert out.shape == (4, 2)

    def test_points_before_the_first_frame(self):
        out = fill_dead_points([5.0, 6.0], [2, 3], n_dead=4)
        np.testing.assert_array_equal(out, [5, 5, 5, 6])

    def test_none_complete(self):
        out = fill_dead_points([1.0, 2.0, 3.0], [0, 1, 2], 3, fill="none")
        np.testing.assert_array_equal(out, [1, 2, 3])

    def test_none_incomplete(self):
        with pytest.raises(ValueError, match="2 of the 4 dead points"):
            fill_dead_points([1.0, 2.0], [0, 2], 4, fill="none")

    def test_frame_outside_the_run(self):
        with pytest.raises(ValueError, match="same run"):
            fill_dead_points([1.0, 2.0], [0, 9], 5)

    def test_duplicate_frames(self):
        with pytest.raises(ValueError, match="same dead point"):
            fill_dead_points([1.0, 2.0], [1, 1], 5)

    def test_unknown_fill(self):
        with pytest.raises(ValueError, match="fill"):
            fill_dead_points([1.0], [0], 1, fill="mean")


SNAPSHOT_MOVES = [
    {"type": "gmc", "species": ["Li"]},
    {
        "type": "gallery_hop",
        "species": ["Li"],
        "host_species": ["C"],
        "in_plane_repeats": [3, 3],
    },
    {
        "type": "layer_slide",
        "species": ["C"],
        "host_species": ["C"],
        "in_plane_repeats": [3, 3],
    },
]


class TestLayeredParams:
    def test_from_snapshot(self):
        params = layered_params({"moves": SNAPSHOT_MOVES})
        assert params == LayeredParams(("C",), ("Li",), (3, 3))

    def test_intercalant_from_start_species(self):
        moves = [
            {"type": "layer_slide", "species": "C", "in_plane_repeats": [6, 6]}
        ]
        cfg = {"moves": moves, "init": {"start_species": "6 36 3 6"}}
        params = layered_params(cfg)
        assert params == LayeredParams(("C",), ("Li",), (6, 6))

    def test_no_layered_move(self):
        with pytest.raises(ValueError, match="no layered move"):
            layered_params({"moves": [{"type": "gmc"}]})


def _atoms(spacings, li, it=None, tilt=None):
    pos, typ, cell = graphite(spacings, li)
    if tilt is not None:
        cell[2, :2] = tilt
    atoms = ase.Atoms(
        [SYMBOLS[t] for t in typ], positions=pos, cell=cell, pbc=True
    )
    if it is not None:
        atoms.info["iter"] = it
    return atoms


def _monitor(n_dead, live, *, with_configs=True):
    """A Monitor with ``n_dead`` dead points and the ``live`` Atoms as live
    walkers."""
    kw = {}
    if with_configs:
        kw = dict(
            live_positions=np.stack([a.positions for a in live]),
            live_cells=np.stack([a.cell[:] for a in live]),
            live_types=np.stack(
                [
                    [LI if s == "Li" else C for s in a.get_chemical_symbols()]
                    for a in live
                ]
            ),
        )
    return Monitor(
        dead_energies=-np.arange(n_dead, dtype=float),
        dead_volumes=None,
        live_volumes=None,
        live_energies=-n_dead - np.arange(len(live), dtype=float),
        log_evidence=0.0,
        iteration=n_dead,
        n_live=len(live),
        n_cull=1,
        symbol_map=SYMBOLS,
        label="test",
        **kw,
    )


FRAMES = [
    _atoms([3.4, 3.4], (), it=0),
    _atoms([3.7, 3.35], (3,), it=3, tilt=(1.42, 0.0)),
    _atoms([3.7, 3.7], (3, 3), it=6),
]
LIVE = [_atoms([3.75, 3.7], (3, 3)), _atoms([3.6, 3.6], (3, 3))]


class TestLayeredObservables:
    def test_dead_then_live(self):
        mon = _monitor(8, LIVE)
        obs = layered_observables(mon, FRAMES, PARAMS)
        assert obs["gallery_width"].shape == (10, 2)
        assert obs["h"].shape == (10,)
        # Dead points: each frame held up to the next one.
        np.testing.assert_allclose(
            obs["h"][:8], [6.8] * 3 + [7.05] * 3 + [7.4] * 2
        )
        np.testing.assert_array_equal(
            obs["n_empty"][:8], [2] * 3 + [1] * 3 + [0] * 2
        )
        # Live walkers: the same values as one configuration at a time.
        for k, atoms in enumerate(LIVE):
            ref = configuration_observables(
                atoms.positions,
                atoms.cell[:],
                atoms.get_chemical_symbols(),
                PARAMS,
            )
            for name, value in ref.items():
                np.testing.assert_allclose(
                    obs[name][8 + k], value, equal_nan=True
                )

    def test_values_feed_the_monitor(self):
        mon = _monitor(8, LIVE)
        obs = layered_observables(mon, FRAMES, PARAMS)
        T = np.array([0.5, 2.0])
        assert mon.expectation(obs["gallery_width"], T).shape == (2, 2)
        assert mon.class_probability(obs["n_empty"] == 1, T).shape == (2,)

    def test_no_live_configurations(self, caplog):
        mon = _monitor(8, LIVE, with_configs=False)
        with caplog.at_level(logging.WARNING):
            obs = layered_observables(mon, FRAMES, PARAMS)
        assert obs["h"].shape == (8,)
        assert "dead points only" in caplog.text

    def test_fill_none_needs_every_dead_point(self):
        with pytest.raises(ValueError, match="fill='hold'"):
            layered_observables(_monitor(8, LIVE), FRAMES, PARAMS, fill="none")

    def test_broken_sheet_names_the_frame(self):
        broken = _atoms([3.7, 3.7], (3, 3), it=6)
        broken.positions[0, 2] += 1.85
        with pytest.raises(ValueError, match="iter=6"):
            layered_observables(
                _monitor(8, LIVE), [*FRAMES[:2], broken], PARAMS
            )

    def test_layout_fixed_by_the_first_frame(self):
        three = _atoms([3.5, 3.5, 3.5], (), it=6)
        with pytest.raises(ValueError, match="expected 2"):
            layered_observables(
                _monitor(8, LIVE), [*FRAMES[:2], three], PARAMS
            )


def _write_run(out, n_dead=8, n_runs=2):
    """An NS output dir: checkpoint, .energies and trajectory per replica,
    config snapshot."""
    out.mkdir(parents=True, exist_ok=True)
    live = np.stack([[a.positions for a in LIVE]] * n_runs)
    cells = np.stack([[a.cell[:] for a in LIVE]] * n_runs)
    types = np.array(
        [
            [LI if s == "Li" else C for s in a.get_chemical_symbols()]
            for a in LIVE
        ]
    )
    with h5py.File(out / "ns.final.checkpoint.h5", "w") as f:
        f.create_dataset("positions", data=live)
        f.create_dataset("types", data=np.stack([types] * n_runs))
        f.create_dataset("cells", data=cells)
        f.create_dataset("energies", data=np.full((n_runs, 2), -100.0))
        f.create_dataset("log_evidence", data=np.zeros(n_runs))
        f.create_dataset("iteration", data=np.full(n_runs, n_dead))
        f.create_dataset("n_dead", data=np.full(n_runs, n_dead))
        f.attrs["n_walkers"] = 2
        f.attrs["symbol_map"] = json.dumps(SYMBOLS)
    for i in range(n_runs):
        elog = EnergyLogger(out / f"ns.run{i:02d}.energies", n_walkers=2)
        elog.write_header()
        for k in range(n_dead):
            elog.write_entry(k, energy=float(-k), volume=100.0)
        elog.close()
        ase.io.write(out / f"ns.run{i:02d}.traj.extxyz", FRAMES)
    snapshot = {"output": {"out_file_prefix": "ns"}, "moves": SNAPSHOT_MOVES}
    (out / "ns.config.snapshot.yaml").write_text(yaml.safe_dump(snapshot))


class TestLoadLayeredObservables:
    def test_every_replica(self, tmp_path):
        _write_run(tmp_path)
        coll, obs = load_layered_observables(tmp_path)
        assert len(coll) == len(obs) == 2
        for mon, o in zip(coll, obs, strict=True):
            assert o["gallery_width"].shape == (mon.n_points, 2)
            np.testing.assert_array_equal(o["gallery_occupancy"][-1], [3, 3])
        assert (tmp_path / "ns.run00.traj.obs.npz").exists()

    def test_cache_reused(self, tmp_path, monkeypatch):
        _write_run(tmp_path)
        _, first = load_layered_observables(tmp_path)

        def fail(*args, **kwargs):
            raise AssertionError("cache not used")

        monkeypatch.setattr(layered_run, "layered_observables", fail)
        _, again = load_layered_observables(tmp_path)
        for a, b in zip(first, again, strict=True):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])

    def test_cache_recomputed_when_the_trajectory_changes(self, tmp_path):
        _write_run(tmp_path)
        _, first = load_layered_observables(tmp_path)
        changed = [*FRAMES[:2], _atoms([3.8, 3.8], (3, 3), it=6)]
        ase.io.write(tmp_path / "ns.run00.traj.extxyz", changed)
        _, again = load_layered_observables(tmp_path)
        assert again[0]["h"][7] == pytest.approx(7.6)
        assert first[0]["h"][7] == pytest.approx(7.4)
