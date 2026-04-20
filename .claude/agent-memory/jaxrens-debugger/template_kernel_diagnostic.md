# Template: single-kernel diagnostic script

Scaffold for `/tmp/diag_<move>.py`. Exercises one move kernel in isolation on a realistic walker state, classifies rejects.

## Template

```python
"""Diagnostic: invoke the <MOVE> kernel at varying step sizes on a clean walker."""
import jax
import jax.numpy as jnp

from jaxrens.backends.lj import create_lj
from jaxrens.backends.ensemble import EnsembleBackend, make_ensemble_params
from jaxrens.sampling.moves.<MOVE> import build_kernel as build_move
from jaxrens.init.positions import grid_positions_in_cell
from jaxrens.init.cells import sample_initial_volume, cell_shape_walk
from jaxrens.utils.cell import get_volume
from jaxrens.state.mc_state import make_mc_state_class

# === Configure ===
n_atoms = 64
max_vol_per_atom = 20.0
min_vol_per_atom = 0.5
min_aspect = 0.6
pressure = 0.1
step_sizes = [0.3, 0.1, 0.01, 1e-3, 1e-5]

# === Build clean walker state ===
key = jax.random.key(42)
k_v, k_c, k_p = jax.random.split(key, 3)

lc = float(sample_initial_volume(k_v, n_atoms, max_vol_per_atom, False))
cell0 = jnp.eye(3) * lc
cell, _ = cell_shape_walk(k_c, cell0, 50, 0.1, 0.1, min_aspect,
                          n_atoms, max_vol_per_atom, min_vol_per_atom)
positions = grid_positions_in_cell(k_p, cell, n_atoms, 1.0)
types = jnp.zeros(n_atoms, dtype=jnp.int32)

lj = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
backend = EnsembleBackend(lj, pressure=pressure)
ep = make_ensemble_params(pressure=pressure)

E0, _, _ = backend(positions, types, cell, 0, ensemble_params=ep)
print(f"Initial H={float(E0):.3f}, V/atom={float(get_volume(cell))/n_atoms:.3f}")

# === Build MCState ===
MC = make_mc_state_class({})
state_base = dict(
    positions=positions, types=types, energy=E0, cell=cell,
    step_size=jnp.asarray(0.0), step_sizes=jnp.full(4, 0.0),
    n_accepted=jnp.zeros(4, dtype=jnp.int32),
    n_proposed=jnp.zeros(4, dtype=jnp.int32),
    max_neighbor_count=jnp.asarray(0, dtype=jnp.int32),
    overflow=jnp.asarray(False), ensemble_params=ep,
)

# === Build kernel ===
move_kernel = build_move(
    backend, n_atoms=n_atoms,
    max_vol_per_atom=max_vol_per_atom, min_vol_per_atom=min_vol_per_atom,
    min_aspect=min_aspect,
    # flat_v_prior=False,  # volume only
)

# emax = E0 + generous headroom; in NS this would be the worst-walker H
emax = float(E0) + 100.0

# === Scan step sizes ===
for ss in step_sizes:
    n_trials = 200
    n_by_reason = [0, 0, 0, 0]
    state = MC(**{**state_base, "step_size": jnp.asarray(ss)})
    kk = jax.random.key(0)
    for _ in range(n_trials):
        kk, k = jax.random.split(kk)
        _, info = move_kernel(k, state, emax)
        n_by_reason[int(info.reject_reason)] += 1
    acc = n_by_reason[0] / n_trials
    print(f"  ss={ss:.0e}  acc={acc:.3f}  "
          f"E_rej={n_by_reason[1]/n_trials:.3f}  "
          f"cell_rej={n_by_reason[2]/n_trials:.3f}  "
          f"prior_rej={n_by_reason[3]/n_trials:.3f}")
```

## Instructions for use

1. Replace `<MOVE>` with the target (`volume`, `shear`, `stretch`, `galilean`, `random_walk`, etc.).
2. Adjust `n_atoms`, `pressure`, `max_vol_per_atom`, etc. to match the user's config.
3. For non-cell moves (galilean, random_walk): drop the cell-constraint kwargs in `build_move(...)` and set `p_accept=1.0` if the move has no prior factor.
4. Run:
   ```
   /home/nunglert/miniconda3/envs/jaxrens/bin/python /tmp/diag_<move>.py
   ```
5. Read output:
   - Healthy kernel: `acc=1.00` at smallest `ss`, dropping smoothly as `ss` increases.
   - Broken kernel: `acc=0.00` at all step sizes → the kernel itself is bugged or the setup doesn't actually produce a valid walker.
   - C-rejections dominant at moderate ss: volume or aspect constraint tight.
   - E-rejections dominant at tiny ss: walker is pathologically at emax boundary (re-check cell/positions construction).
6. Delete `/tmp/diag_<move>.py` when done.

## When NOT to use this template

- If the user's symptom is clearly about multi-step chain behavior (rather than single-move acceptance), build a different diagnostic that runs `n_mcmc_steps` of `step_fn` via `jax.lax.scan` and measures cumulative acceptance.
- If the symptom is cross-run / cohort / vmap related, the isolation target is the resolver or `run_ns_parallel`, not a single kernel.
