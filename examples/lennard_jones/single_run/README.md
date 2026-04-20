# LJ-8 NPT

8-atom Lennard-Jones solid in the NPT ensemble at P = 0.1 (reduced units).

## What this example exercises

- **Mixed-move MWG**: Galilean atomic moves + volume + shear + stretch.
- **NPT ensemble**: `EnsembleBackend` wraps the LJ backend with a PV term.
- **Periodic LJ**: `cutoff = 2.5σ`.
- **Random initialization** with grid-placement and cell-shape walk.
- **Burn-in**: 5 walks × 50 steps at fixed Emax before NS proper, to
  decorrelate walkers from their grid initialization.
- **Step-size adaptation** (`full_auto: true`) with a 0.30–0.70 target window.
- **Termination** on either 5000 iterations or prior-mass < 1e-3.
- **Trajectory + checkpoint output** every 50 / 500 iterations.

## Running

```bash
cd experiments/examples/lj8_npt
python -m jaxrens.cli.cli validate -c config.yaml    # sanity check
python -m jaxrens.cli.cli run -c config.yaml         # the run itself
```

Artifacts land under `./output/`:
- `lj8_npt.traj.extxyz` — trajectory (culled walker per N iterations).
- `lj8_npt.energies` — energy log.
- `lj8_npt.checkpoint.h5` — HDF5 checkpoint for restart.

## Resuming a crashed run

```bash
python -m jaxrens.cli.cli run -c config.yaml \
    --set init.start_species=null \
    --set init.restart_file=./output/lj8_npt.checkpoint.h5
```

(Restart path skips burn-in automatically.)

## Overriding parameters at the CLI

```bash
# Smaller smoke-test run:
python -m jaxrens.cli.cli run -c config.yaml \
    --set run.n_live=64 --set run.max_iterations=200

# Different pressure:
python -m jaxrens.cli.cli run -c config.yaml --set ensemble.pressure=1.0
```

## Expected outcome

`log_evidence` should decrease monotonically as the run descends through
liquid-like and ordered configurations. The live-walker energy trace
(visible via `info_interval=100` log lines) should eventually cluster
around the LJ FCC minimum near `E/atom ~ -6` (reduced units) for this
small cluster under moderate pressure.

Runtime on a single modern GPU: a few minutes to ~10 min, depending on
compilation warmup and convergence speed.

## Scaling up

For a more demanding NS calculation:

- Increase `run.n_live` (512–2048 is the canonical range for LJ-NS).
- Increase `run.n_mcmc_steps` (50–200 for better decorrelation).
- Raise `run.max_iterations` or lower `termination.prior_mass.threshold`.
- Set `init.initial_walk.walker_batch_size` if walker vmap exceeds device
  memory for larger systems.

## Changing the pressure sweep

A cohort over multiple pressures runs each sequentially (for now):

```yaml
ensemble:
  type: npt
  pressure: [0.01, 0.1, 1.0]
  pressure_units: eva3
```

`jaxrens validate` will report `cohort size: 3`.
