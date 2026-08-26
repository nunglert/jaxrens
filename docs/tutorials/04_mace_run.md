# 5. MACE Sn — MLIP

A real production config from an active-learning nested-sampling campaign:
16 atoms of tin (Sn), a MACE potential, and 32 replicas spanning a pressure
ladder from 0 to 15.5 GPa with pressure-RENS exchange between them. This is
not something to run casually — to convergence it is a multi-GPU job
measured in hours — so this page reads the config and validates it rather
than running it. Every choice below is the kind you make once you are past
"does the loop work" and into "will this result hold up."

## The config

`examples/tutorials/04_mace_sn16/config.yaml`:

```{literalinclude} ../../examples/tutorials/04_mace_sn16/config.yaml
:language: yaml
```

## Why these parameters

### `interval_units: per_walker`

Every interval field in this config — `adaptation.adjust_interval`,
`inter_re.re_interval`, every `output.*_interval`, even
`run.max_iterations` — is in **walker-sweeps**, where one sweep is
`run.n_live` iterations, not raw iteration counts. The resolver multiplies
each by `n_live` before building the runtime dataclasses. The point is
scale invariance: `adjust_interval: 0.5` means "twice per sweep" whether
`n_live` is 32 or 2000, so tuning `n_live` later does not silently retune
every cadence in the file along with it. `run.max_iterations: 4000` reads
as 4000 sweeps — 800,000 raw iterations once resolved (`jaxrens validate`
below shows exactly that number) — a ceiling deliberately high enough that
`termination:` should always fire first.

### Backend: `supercell_trafo: [3, 3, 3]`

Same arithmetic as {doc}`03_lj_npt`'s cutoff check, applied to a
message-passing model instead of a pairwise cutoff: the tiled cell has to
span at least twice the model's receptive field at the *tightest* cell
`cell:` below permits, not at some typical volume. A prior floor of
`min_volume_per_atom: 18.0` over 16 atoms is nowhere near as extreme as
{doc}`03_lj_npt`'s close-packed floor, but a receptive field several
Angstrom wide still needs more than the default `[2, 2, 2]` tiling to stay
covered across the whole prior — hence `[3, 3, 3]`.

### Backend: `max_neighbors_list`, `max_neighbors_offset`, `max_neighbors_shrink_dwell`

MACE is a message-passing GNN, not a fixed-cutoff pair potential like
{doc}`03_lj_npt`'s LJ backend: the energy needs an explicit neighbor graph,
and `jax.jit`'s static-shape rule means the edge buffer has to be
preallocated at some fixed per-atom capacity — `max_neighbors` — before any
actual geometry is known. See {doc}`/user/concepts/backends`'s "The
neighbor problem" for the full escalate/recompile mechanism; the three
knobs here are its user-facing surface:

- `max_neighbors_list: [20, 25, 30, 35, 40, 45, 50, 60]` is the allowed
  bucket ladder. The resolver picks the smallest entry that fits the
  starting geometry before the run begins; if a later MCMC step pushes the
  true neighbor count past the current bucket, the outer loop rolls that
  one step back, escalates to the next ladder entry, and retries — one JIT
  recompile per distinct bucket, cached for the rest of the run. Eight
  close-spaced entries bound the whole 800,000-iteration run to at most
  eight recompiles.
- `max_neighbors_offset: 4` is the headroom added to the observed peak when
  picking a bucket, so a small fluctuation right after growing doesn't
  immediately trip another escalation.
- `max_neighbors_shrink_dwell: 10` lets the bucket shrink back down too —
  after 10 consecutive iterations comfortably below the next-smaller entry,
  the run steps down to it (reusing that bucket's already-compiled kernel).
  Without this the ladder only ever grows.

The shrink side matters more here than it would for a single-replica run:
`max_neighbors` is one static field shared by the *entire* batched
population passed to one JIT-compiled kernel, so with all 32 pressure
replicas batched together, every replica runs at whichever bucket the
*single tightest* replica currently needs — even the 0 GPa replica sitting
in a loose, sparse configuration pays for the 15.5 GPa replica's dense one.
Cell moves are exactly what drives this: with `ensemble.type: npt` (see
{doc}`/user/concepts/backends`'s "Ensembles as additive corrections"),
`volume`/`stretch`/`shear` are live moves, and every one of them changes
the cell — which is what changes the true neighbor count, on top of
whatever plain atomic motion does. Without `shrink_dwell`, one compressed
excursion early in the run would pin every replica at the largest bucket
it ever needed for the rest of it; with it, the ladder tracks the current
tightest replica instead of the tightest one so far.

### `ensemble.pressure`: 32 replicas, not a handful

One replica every 0.5 GPa from 0 to 15.5. This dense a ladder only pays for
itself with `inter_re` doing real work between adjacent rungs — see below —
and it is what turns a single P-V curve into a resolved equation of state
across a pressure range wide enough to cross a phase transition, rather
than three or four isolated points that might straddle one without ever
landing near it.

### `inter_re.re_interval: 0.002` — as aggressive as the clamp allows

0.002 sweeps at `n_live: 200` is 0.4 raw iterations; the resolver clamps
every scaled interval to at least 1, so this fires **every single NS
iteration**. `jaxrens validate` below reports it as an advisory warning —
"values below 0.05 are unusual" — and normally that heuristic is right: it
is guarding against a leftover raw-iteration count that got misread as
sweeps. Here it does not apply. A 32-replica pressure ladder only resolves
sharp features if walkers actually migrate across it, and with adjacent
rungs only 0.5 GPa apart, swap attempts are cheap to offer relative to an
MLIP energy evaluation. Firing every iteration is the deliberate choice,
not the mistake the warning is written to catch.

### Moves: `gmc` *and* `random_walk` together

Both move atoms, and that is deliberate, not redundant. `gmc` uses the
model's forces and is efficient wherever they are smooth; `random_walk`
does not depend on gradient quality at all, which matters for a
message-passing potential whose forces can be noisy near its cutoff or in
under-explored geometries. Keeping both gives the sampler a fallback that
does not share `gmc`'s failure mode. Cell moves (`volume`, `stretch`,
`shear`) still carry most of the weight — 32 of 44 — same as any NPT run.

### `adaptation.per_move`: keyed by move **name**, not the `galilean` alias

`type: galilean` in the `moves:` list is rewritten to `gmc` at parse time —
but that alias is *only* applied to the moves list. `adaptation.per_move`
is keyed by move name, which falls back to `type:` when no explicit `name:`
is set, so a `per_move: {galilean: ...}` block would silently miss the
`gmc` move entirely and it would fall through to `defaults` instead. This
config spells it `gmc:` for exactly that reason.

### `termination: temperature, target_temp: 100.0`

Stop on physics, not on an iteration count: once the finite-difference
temperature estimate cools to 100 K, the run has resolved what it set out
to. `run.max_iterations` is the safety net underneath that, not the thing
actually controlling when the run ends.

### `cell.{min,max}_volume_per_atom`: bracket a real density, not a guess

beta-Sn's bulk density corresponds to roughly 27 Å³/atom. `18.0`–`90.0`
brackets that with real headroom on both sides — compressed through
expanded/molten — rather than a symmetric-looking range picked without
reference to the material the potential is supposed to describe.

### `init.pos_randomization_mode: grid`

This is the default, made explicit here rather than left implicit. Each
walker's 16 starting atoms are placed on a regular lattice spaced by
`grid_distance: 3.0` Å, with 16 of those sites chosen at random — as
opposed to `pos_randomization_mode: uniform`, which draws each atom's
position independently and uniformly inside the cell. Every pair of grid
sites is at least `grid_distance` apart *by construction*; a uniform draw
carries no such guarantee and will, occasionally, place two atoms almost
on top of each other.

That distinction matters specifically for a foundation MLIP. A
uniform-mode near-collision is normally caught after the fact by
`init.start_energy_ceiling_per_atom` — reject the whole configuration if
its energy is absurd — which works fine for a potential with an explicit
short-range repulsive term, since the energy really does diverge as atoms
approach and the ceiling reliably fires. MACE and other foundation models
give no such guarantee: as the `constraints` section right below notes,
they are not always well-behaved at very short range, and a configuration
close enough can land in geometry the model was never
trained on and report a spuriously low energy instead of a diverging one —
an artificial minimum the ceiling has no way to catch, because nothing
about the reported number looks wrong. `grid` avoids the region
structurally rather than statistically: the starting population simply
never visits short range in the first place, independent of what any
energy-based check downstream would or wouldn't have caught.

### `constraints: minimum_distance, d_min: 1.7`

Foundation MLIPs are not always well-behaved at very short range — without
a proper short-range repulsive term, walkers can find configurations the
model was never trained on and reports as spuriously low-energy. A hard
floor at 1.7 Å is cheap insurance against exactly that, independent of
whatever the model's forces say near contact. `init.pos_randomization_mode:
grid` above is the same defense at initialization time, before sampling
ever starts; this constraint is what keeps every MCMC step afterward
honest too.

### Output: debug logging and dense trajectory output, on purpose

`log_level: debug`, `save_re_stats: true`, `save_max_neighbors: true`, and
`traj_interval: 0.02` (every ~4 iterations — the second advisory warning
`jaxrens validate` reports below) all trade real disk I/O and log volume
for maximum diagnostic fidelity. That trade only makes sense for a run
whose trajectory feeds something downstream — here, active-learning
retraining, which is what the checkpoint path this config was captured
from was for. A one-off exploratory run would not want this much output.

## Validate it

```bash
cd examples/tutorials/04_mace_sn16
jaxrens validate -c config.yaml
```

```text
UserWarning: inter_re.re_interval=0.002 with interval_units=per_walker fires
~500x per walker-sweep, which is rarely intended and very expensive. ...
UserWarning: output.traj_interval=0.02 with interval_units=per_walker fires
~50x per walker-sweep, which is rarely intended and very expensive. ...
✓ OK — configuration plan valid
  topology  n_gpu=1 × n_per_gpu=32 = 32 replica(s)
  run       n_live=200, max_iterations=800000
  moves     5 move(s) [gmc, random_walk, volume, stretch, shear]
  backend   mace, n_atoms=16
  output    format=extxyz, prefix=mace_sn16
```

Both warnings are the ones discussed above — real, expected, and
deliberately not the mistake they usually flag. `max_iterations=800000` is
`4000 × n_live` exactly, confirming the sweep arithmetic. `--full` — which
would build the backend, place all 32 replicas' walkers and evaluate their
initial energies — is deliberately not run here: with a real checkpoint it
needs enough memory to hold 32 replicas of a loaded MACE model, which is a
GPU-sized job, not something to do from a laptop between commands. Point
`backend.checkpoint_path` at your own converted model and run `--full` on
real hardware before queuing the real thing.

## Next

- {doc}`03_lj_npt` — the same shape without the scale: a single replica,
  a cheap backend, and validate output you actually run.
- {doc}`/user/concepts/backends` — the neighbor-bucket escalation mechanism
  and the per-ensemble energy terms in full.
- {doc}`/user/mace_models` — converting a torch MACE checkpoint into the
  bundle format `checkpoint_path` expects.
- {doc}`/user/concepts/replicas` — how the topology below `ensemble.pressure`
  is derived, and what a swap actually does.
- {doc}`/reference/config` — the `inter_re:` and `ensemble:` surfaces in
  full.
