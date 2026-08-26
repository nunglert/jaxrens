# 4. LJ periodic — NPT

The same 8 atoms as {doc}`02_lj_cluster`, with the cell back: periodic
images, pressure, and the moves that act on a cell. Unlike the quickstart
tutorials this one is not meant to be run casually — a physically sound
result takes a few minutes on one GPU, not seconds — so this page reads
the config and validates it rather than running it. Everything below
transfers directly to a real job: swap the backend for a real potential and
this is the shape a serious periodic NPT run takes.

## The config

`examples/tutorials/03_lj_npt/config.yaml`:

```{literalinclude} ../../examples/tutorials/03_lj_npt/config.yaml
:language: yaml
```

## Why these parameters

**`cell.min_volume_per_atom: 1.2`** is the one that actually decides what
this run can see. An LJ solid's close-packed volume is about 1.0 σ³/atom
(FCC nearest-neighbour distance `2^(1/6) σ`), so a prior floor left at a
demo-comfortable value — 10, 20, whatever avoids thinking about it — never
lets the sampler anywhere near the solid phase; it would spend the whole run
exploring gas-like configurations and report a nested-sampling result that
quietly never touched the physics anyone actually wants. 1.2 sits just above
close-packed, with a little headroom so walkers are not pinned exactly at a
hard boundary throughout the run.

**`backend.supercell_trafo: [4, 4, 4]`** follows from that floor, not from
habit. The cutoff-vs-cell-prior check has to hold at the *tightest* legal
cell, not at some typical volume: `worst_perpendicular_distance ×
min(supercell_trafo) ≥ 2 × cutoff`. At `min_volume_per_atom: 1.2` and
`min_aspect_ratio: 0.7` the worst-case perpendicular distance is `(1.2 × 8
atoms)^(1/3) × 0.7 ≈ 1.49 Å` — so `[2, 2, 2]`, which is fine for a loose,
gas-like prior, falls well short of the required 5 Å here, and only `[4, 4,
4]` clears it. `jaxrens validate` below shows this directly: drop the floor
back to a demo-loose value and `[2, 2, 2]` stops warning; keep the floor
where the physics needs it and the warning is telling you something real.

**`run.n_live: 200`**, up from a demo's 32: more live points means less
Monte Carlo noise in the log-evidence and in anything computed from it
(`jaxrens analyze`'s heat capacity, free energy, …), and 8 atoms is cheap
enough that 200 costs nothing to justify.

**`run.n_extra: 15` and `run.n_mcmc_steps: 20`** set two different things,
easy to conflate since both sound like "more sampling":

- `n_extra` is the **parallel batch width** — each outer NS iteration walks
  `1 + n_extra` walkers side by side under one `vmap`: the walker that just
  replaced the culled one, plus `n_extra` further survivors resampled from
  the live population to decorrelate them too. It does not enlarge the
  population (`n_live` alone governs that) and the extras never become dead
  points; `0` leaves a GPU walking a single chain and mostly idle. At `15`
  that's 16 walkers walked per iteration.
- `n_mcmc_steps` is the **chain length** — how many sequential move
  proposals each of those 16 walkers takes, one weighted draw from
  `{gmc, volume, shear, stretch}` per step.

So this config proposes `(1 + n_extra) × n_mcmc_steps = 16 × 20 = 320`
moves per outer NS iteration — a number fixed by the config, independent of
which moves actually get drawn. What that costs in backend evaluations is
not fixed: `volume`, `shear`, and `stretch` each spend one evaluation per
proposal, but `gmc` runs its own `n_reflect`-step trajectory internally
(`n_reflect: 6` here) — every reflection is a full energy-and-force call, so
one gmc proposal costs 6 evaluations, not 1. With gmc drawn at weight `4`
out of `4 + 1 + 1 + 1 = 7`, the *expected* cost per iteration is roughly
`320 × (4/7 × 6 + 3/7 × 1) ≈ 1200` evaluations — an estimate, not an exact
count, since which move gets drawn each step is random. The actual running
total is what a live run's `nE`/`nG` counters report (see the printed log
in {doc}`02_lj_cluster`, or `jaxrens plot`'s energies output), not
something to hand-derive from the YAML alone.

**`termination: prior_mass, threshold: 1.0e-5`**, tighter than a demo's
`1.0e-3`: the run stops when the remaining prior mass genuinely cannot move
`log_Z` further, not after an arbitrary iteration count. `run.max_iterations:
200000` is a generous ceiling underneath that — high enough it should never
actually be hit — not the thing controlling when the run ends.

**`constraints: minimum_distance, d_min: 0.8`** is cheap insurance on top of
the LJ repulsive core: a hard floor that rejects a proposal landing two atoms
on top of each other before the potential's own repulsion gets a chance to.

## Validate it

```bash
cd examples/tutorials/03_lj_npt
jaxrens validate -c config.yaml --full
```

```text
✓ OK — configuration valid
  topology  SingleRun (1 replica, 1 GPU)
  run       n_live=200, max_iterations=200000
  moves     4 move(s) [gmc, volume, shear, stretch]
  backend   lj, n_atoms=8
  output    format=extxyz, prefix=lj8
```

`--full` is worth the extra second here — beyond schema checks it builds the
backend, places the walker population under `cell:`'s bounds, and evaluates
their initial energies, so a broken model file or an unsatisfiable cell
prior surfaces now rather than after a job has queued.

To see the cutoff check actually fire, loosen the floor back toward a
demo-comfortable value and watch `[2, 2, 2]` stop being enough:

```bash
jaxrens validate -c config.yaml \
    --set cell.min_volume_per_atom=4.0 \
    --set backend.supercell_trafo=[2,2,2]
```

```text
LJ cutoff vs cell-prior bounds: smallest legal cell has worst-case
perpendicular distance 2.2224 A ... below the required 2 * cutoff = 5.0000 A.
LJ energies will undercount neighbours on the tight end of the cell-prior
range.
```

That configuration would actually validate — the warning is advisory, not
fatal — but it is exactly the trap described above: a comfortable-looking
prior that never reaches the density where the interesting physics lives.

## Next

- {doc}`04_mace_run` — the same shape at production scale: a real
  interatomic potential, a 32-pressure replica ladder, and the parameter
  choices that go with running an active-learning campaign for real.
- {doc}`/reference/config` — every key you can put in that YAML.
- {doc}`/user/concepts/ns_loop` — what the two loops are actually doing.
