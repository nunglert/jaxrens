# Troubleshooting

Common problems and the knobs that fix them. Each entry names the symptom, the
cause, and the specific YAML fields (or environment variables) to change.

## Out-of-memory (OOM) errors

GPU OOM (`RESOURCE_EXHAUSTED`, `XlaRuntimeError: out of memory`) is the most
common failure, and it's almost always driven by **how many walkers are
evaluated in parallel**, not by the size of any single stored array. jaxrens
`vmap`s the backend (energy/forces + neighbor graph) across a whole batch of
walkers at once, so the peak footprint is the per-walker working memory
**times the batch width** — every walker's intermediate buffers are live on
the device simultaneously. The stored population itself is comparatively
small; it's this transient parallel evaluation that blows the budget.

The fix is therefore to **narrow that batch**.

| Where it OOMs | Knob | Section | Effect |
|---|---|---|---|
| Burn-in / initial walk | **`walker_batch_size`** | `init.initial_walk` | Chunks the per-walker vmap via `lax.map` instead of vmapping all walkers at once. Set to e.g. `8` or `4`; smaller = less memory, slightly slower. `None` (default) vmaps everything. |
| NS main loop | **`n_extra`** | `run` | Extra walkers cloned + walked each iteration. Each one is a full backend evaluation held in memory — drop it toward `0`. |
| Step-size adaptation (bisection) | **`trial_batch_size`** | `adaptation` | Each bisection round vmaps a batch of **trial** walkers; this chunks that vmap via `lax.map` (the adaptation analog of `walker_batch_size`). Defaults to `8`; lower it further if adaptation is where you OOM, or set `None` to vmap all trials at once. Alternatively lower **`adjust_n_samples`** (trial walkers per round, default `50`) to shrink the batch itself.

## Neighbor-buffer overflow

**Symptom:** repeated recompilation mid-run, or a log line about the bucket
being escalated on nearly every iteration.

**Cause:** `max_neighbors` (the per-atom edge budget) is too small for the
live geometry, so the outer loop keeps bumping to the next bucket — each new
bucket is a fresh JIT compile.

**Fix (`backend` section):**
- Widen or raise `max_neighbors_list` so a comfortable bucket exists.
- Raise `max_neighbors_offset` (the headroom added when picking the initial
  bucket) so a single MCMC step doesn't immediately re-trip the overflow.

This is expected behaviour a *few* times early in a run; it's only a problem if
it never settles.  
This occurs frequently if the potential shows artificial attraction at low bond distances. Some foundation models without proper short range repulsion component show this kind of failure, which manifests in clumped configurations with very unphysical neighbor counts.  
For the full mechanism, see {doc}`concepts/backends`.

## NaN energies / walkers collapsing

**Symptom:** energies go to `NaN` or `-inf`; walkers pile atoms on top of each
other; with MACE the step size collapses and the run stalls.

**Cause:** MLIPs have no defined behaviour for close-contact geometries far
outside their training set — the learned potential can turn attractive or
`NaN`. Under high pressure or aggressive cell moves, walkers drift into that
region and irreversibly collapse.

**Fix:** enable the soft-core repulsion wrapper on the backend
(`backend.softcore_repulsion`), which adds a parameter-free repulsive Morse
term that goes to `+∞` at short range:

```yaml
backend:
  type: mace
  checkpoint_path: ...
  softcore_repulsion:
    r_core_cut: 1.25      # Å; repulsion active below this
    r_core_switch: 0.75   # Å; must be < r_core_cut
```

Tune `r_core_cut` to the physically shortest bond you expect. Setting it *too*
large can itself jam walkers against an artificial wall (they get stuck at
`r_core_cut` instead of exploring), which can masquerade as a freezing
transition — so start near a real close-contact distance and only raise it if
collapse persists.

**The other tool for the same failure** is a hard minimum-distance
constraint, which rejects any proposal bringing two atoms closer than a floor
you set instead of penalising it energetically:

```yaml
constraints:
  - type: minimum_distance
    d_min: 0.8            # uniform floor (Angstrom)
```

`d_min` also accepts per-species-pair floors (`{default: 1.0, Si-Si: 2.0}`).
The two approaches differ in kind: the soft-core term deforms the potential
everywhere below `r_core_cut`, so it changes the sampled distribution; a
constraint restricts the prior instead, exactly like the likelihood
threshold, and costs nothing where it isn't violated. Reach for the
constraint when you want a hard floor without touching the energetics, and
the soft core when the model itself needs repairing at short range. See
{doc}`/reference/config` for the full constraint surface.

## Still stuck?

- Reproduce with a tiny config (`n_live: 32`, a handful of iterations) to
  isolate whether it's a config, memory, or model problem.
- Check the `<working_dir>/<prefix>.log` file — set `output.log_level: debug`
  for the full trace.
- Open an issue with the config, the traceback, and `nvidia-smi` output at
  <https://github.com/nunglert/jaxrens/issues>.
