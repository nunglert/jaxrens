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

### `constraints: minimum_distance, d_min: 1.7`

Foundation MLIPs are not always well-behaved at very short range — without
a proper short-range repulsive term, walkers can find configurations the
model was never trained on and reports as spuriously low-energy. A hard
floor at 1.7 Å is cheap insurance against exactly that, independent of
whatever the model's forces say near contact.

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
- {doc}`/user/mace_models` — converting a torch MACE checkpoint into the
  bundle format `checkpoint_path` expects.
- {doc}`/user/concepts/replicas` — how the topology below `ensemble.pressure`
  is derived, and what a swap actually does.
- {doc}`/reference/config` — the `inter_re:` and `ensemble:` surfaces in
  full.
