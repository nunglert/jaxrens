# SMC and waste-free SMC on jaxrens — reusability survey

## Context

You are weighing adding **Sequential Monte Carlo (SMC)** and **waste-free SMC** (Dau & Chopin 2022) alongside the existing nested-sampling pipeline. The question is how much of the current infrastructure carries over and how much would have to be written fresh — including the CLI, config schema, monitors, and trajectory I/O. The `fwf_proposal.pdf` in the repo already flags NS-SMC as a planned direction.

Short answer: **the codebase is unusually well-suited to this**. The two-loop architecture (Python outer / `lax.scan` inner) maps almost 1:1 onto SMC's reweight–resample–rejuvenate structure, and waste-free SMC is, mechanically, "keep all the intermediate `lax.scan` states instead of discarding them" — which the existing scan already produces. A single ~15-line plumbing change to `ns_step` is the only real precondition.

---

## Part 1 — The structural fit

SMC iteration = `reweight → (optional) resample → MCMC rejuvenation`. Mapping onto jaxrens:

| SMC step | jaxrens equivalent | Effort |
|---|---|---|
| Reweighting against new target (e.g. tempering β, NS-threshold) | New: `log_w += log L(x) * dβ` (or hard threshold) | New code, ~50 lines |
| ESS computation | New | New code, ~10 lines |
| Resampling (systematic/stratified/multinomial) | New | New code, ~100 lines |
| MCMC rejuvenation of P particles for M steps | **Exactly the existing inner scan** in `ns_step` (`sampling/nested_sampling.py:339-395`) | **Reuse as-is** |
| Waste-free: keep all P×M intermediate states | The scan already produces them; just don't discard | Trivial — change what the scan carries |
| Step-size adaptation between iterations | `adaptation/manager.py` | **Reuse as-is** |

The single non-trivial precondition: `MoveInfo.log_likelihood` is computed by every move kernel but discarded inside `ns_step`'s scan body (`nested_sampling.py:350-377`). Capturing it in the scan carry is ~15 lines and unlocks SMC reweighting. Worth doing regardless of SMC since it's also useful for monitors.

---

## Part 2 — Component-level reuse map

### Reuse as-is (zero or near-zero changes)

- **`sampling/mwg.py`** — the MWG scheduler is algorithm-agnostic. `step_fn(rng, state, likelihood_constraint)` is generic; in NS the constraint is `E_max`, in NS-SMC it's still a threshold, in tempered SMC it's effectively `+inf` (every move proposes; reweighting handles selection). **100% reusable.**
- **`sampling/batch_wrapper.py` + `batch_descriptor.py`** — `SingleRun` / `VmapRuns` / `ShardedSingleRun` accept any step function via the `ns_step_fn` parameter (`run_loop.py:74`). They are pure "wrap a single-walker function into a population-of-populations" plumbing. **99% reusable.**
- **All move kernels** (`sampling/moves/*.py`) — pure MCMC proposals, nothing NS-aware. **100% reusable.**
- **`adaptation/manager.py`** — per-move step-size adaptation is generic to "MCMC inside a population loop". **100% reusable.**
- **Energy backends** (`backends/*`) — algorithm-agnostic by design. **100% reusable.**
- **`state/walker.py` `WalkerState`** — population members are atomistic configurations regardless of the outer algorithm. **100% reusable.**
- **Most config schema** — `MoveConfig`, `BackendConfig`, `OutputConfig`, `InterREConfig`, adaptation specs are all generic. The CLI's YAML parser, `cli/parser.py`, validation, `dump-schema`, `--set` overrides — all algorithm-blind. **~70% reusable.**

### Small surgery (one-line tweaks or interface widening)

- **`sampling/nested_sampling.py` — `ns_step` scan body** (`:339-395`): add `log_likelihood` to the scan carry so SMC can reweight against per-step likelihoods. **~15 lines.**
- **`sampling/termination.py`** — protocols accept `(iteration, emax)` only. Widen to `(iteration, emax, ess=None)` so SMC criteria can read ESS without breaking NS. `IterationTermination` and `EnergyTermination` carry over verbatim; `TempTermination`'s log-quantile math is half-reusable.
- **CLI dispatch** (`cli/run.py`, `cli/cli.py`) — currently hard-imports `run_ns` / `run_ns_parallel` / `run_ns_multi_gpu` / `run_ns_sharded`. Add an `algorithm: "ns" | "smc" | "smc_wf"` field to `NSConfig` (or a new umbrella `RunConfig`) and gate the import. **~30 lines of dispatch.**
- **`cli/resolve.py`** — already builds the descriptor / batcher / adaptation manager / move kernels independently of the outer algorithm. The same builder feeds an SMC runner with no changes to its core. The only addition is constructing SMC-specific objects (resampler, ESS threshold) when `algorithm == "smc"`.

### Write fresh

- **SMC weight update + ESS** — small (≤100 lines combined).
- **Resampling kernels** — multinomial / stratified / systematic. ~150 lines, fully JIT-compatible (standard JAX `jnp.searchsorted` patterns).
- **Adaptive tempering schedule** — bisection to next β that gives target ESS reduction (~50 lines). Trivially JIT-compatible inside `lax.while_loop`.
- **`SMCState`** — parallel to `NSState`: replace `log_evidence` with `log_weights: Float[*B, K]`, add `temperature` / `threshold_idx` / `n_resampling_events`. Same pytree pattern; ~80 lines copy + rename from `state/ns.py`.
- **ESS / weight-based termination criteria** — new `termination.py` entries (~50 lines each).
- **Waste-free–aware trajectory writer** — see Part 4 below. Most of the work is "buffer intermediate scan states then flush"; the existing `write_walker_snapshot` (`io/trajectory.py:45-67`) already supports writing entire populations and would be reused as the underlying operation.
- **SMC monitor** — `TemperatureCallback` in `cli/monitor.py` is NS-specific (consumes `info["emax"]` history). For SMC, replace with ESS / temperature-schedule diagnostics (~150 lines, parallel structure to existing monitor).

---

## Part 3 — Why waste-free SMC is a particularly clean fit

Waste-free SMC's defining trick: after resampling P particles, run each for M MCMC steps and keep **all P×M intermediate states** as the next iteration's particle set (instead of taking only the final state and "wasting" the intermediate ones). This is exactly what jaxrens already produces at the bottom of `ns_step`'s `vmap(lax.scan(n_mcmc_steps))` — the scan already carries every intermediate `WalkerState`; NS just keeps the final one.

Concretely, `nested_sampling.py:339-395`'s `scan_body` would need to:
- Carry the intermediate `WalkerState` (not just the stats), which is one extra entry in the scan's `ys` output (`lax.scan` already returns all intermediate ys).
- Reshape `(P, M, ...)` → `(P*M, ...)` after the scan so the next SMC iteration sees N = P*M particles.

The two-loop boundary is *exactly* where this split lives:
- **Outer (Python):** reweight, resample, choose M, monitor ESS, schedule β, terminate.
- **Inner (JIT):** the existing scan, producing P×M states.

That's the right boundary for waste-free SMC by Dau & Chopin's own framing. No architecture change is required to host this — just an extra return path from the scan.

A subtlety to flag: when M·n_mcmc_kernels-per-step > 1, the P*M particles within one resampling cycle are correlated. That's fine for waste-free SMC (the paper formally accounts for it in the ESS estimator), but the **ESS computation must use the waste-free estimator, not naive Kish ESS**. Worth getting right in the first implementation; it's ~30 lines of math.

---

## Part 4 — CLI, config, and I/O specifics

### CLI surface

Two options for exposing SMC at the CLI:

- **Option α (cheap, recommended):** keep `jaxrens run -c config.yaml` as the entry point; add `algorithm: "ns" | "smc" | "smc_wf"` to the top-level run config. `cli/parser.py` validation stays as one schema; the resolver chooses which runner to instantiate. Backward compatible (default `"ns"`). No new CLI commands; same `--set`, `--set moves[0].step_size=...` overrides work unchanged.
- **Option β (more explicit, more surface to maintain):** new sub-commands `run-ns`, `run-smc`, `run-smc-wf`. Cleaner help text per algorithm but duplicates dispatch boilerplate.

α is the minimum-friction path and matches the existing `inter_re` precedent — a config-flag-driven feature toggle, not a new entry point.

### Config schema

`NSConfig` becomes `RunConfig` (or just keeps the name with new optional fields):

- `algorithm: Literal["ns","smc","smc_wf"] = "ns"`
- Existing `n_live`, `convergence_threshold`, `n_mcmc_steps`, `max_iterations` — keep, repurpose. For SMC, `n_live` is the particle count; `convergence_threshold` becomes an ESS/temperature threshold. Either rename for clarity or keep names and document.
- A new optional `SMCConfig` sub-block with `ess_threshold`, `tempering_target_ess_drop`, `resampling: "multinomial"|"systematic"|"stratified"`, `waste_free: bool`, `n_mcmc_steps_per_iter` (= the M in waste-free).
- `MoveConfig`, `BackendConfig`, `OutputConfig`, adaptation, `InterREConfig` — **unchanged**.

### Output / trajectory I/O

- **`io/trajectory.py`** has both `write_dead_point(iteration, walker, energy)` (NS-specific) and `write_walker_snapshot(iteration, walkers)` (the *entire* live population). The latter is already what SMC needs; it just isn't wired into the default NS callbacks. **Reusable as-is.**
- **`io/energy_log.py` `EnergyLogger`** assumes one energy per iteration (the dead point). For SMC and especially waste-free, the natural granularity is "one row per particle per iteration" or "one row per resampling event". Needs a parallel `WeightLogger` / extended schema. ~100 lines.
- **`cli/monitor.py` `TemperatureCallback`** is NS-only (Baldock-style finite-difference temperature from `E_max` history). For SMC, the right diagnostic is ESS trajectory, β schedule, and acceptance per resampling cycle. Parallel structure, fresh content — ~150 lines.
- **HDF5 / extxyz writers**: the underlying file formats and ASE conversion work for any walker population; only the metadata they tag onto each frame is NS-specific (`dead_point_index`, etc.). Trivial to extend.

---

## Part 5 — Suggested implementation order

1. **Plumb `log_likelihood` through the inner scan** (`ns_step` in `nested_sampling.py:339-395`). ~15 lines. Useful on its own, prerequisite for everything below.
2. **Widen termination protocol** to accept optional `ess` and `weights`. Backward compatible.
3. **`SMCState`** as a sibling of `NSState`. Reuse `WalkerState`, `MCState`, `MoveInfo` verbatim.
4. **Resampling kernels** as a standalone module `sampling/smc/resampling.py`. JIT-compatible.
5. **Adaptive tempering** as `sampling/smc/tempering.py`.
6. **`smc_step`** as a sibling of `ns_step` — reuses the inner MCMC scan loop verbatim by lifting it into a shared utility.
7. **`run_smc`** mirroring `run_ns`'s structure. The outer loop in `run_loop.py:291-540` is largely algorithm-agnostic and may be reusable with a small parameterization; if not, copy + adapt.
8. **Waste-free variant**: a flag on `smc_step` that reshapes the scan's intermediate ys into the new population. Smallest delta on top of standard SMC.
9. **CLI dispatch & config**: add the `algorithm` switch in `cli/resolve.py`.
10. **Monitors and energy/weight logger**: parallel to NS monitors; reuse `write_walker_snapshot` for trajectories.

Rough effort estimate: **~2–3 focused weeks** for standard SMC + waste-free SMC, given the architect-then-implementer workflow the project already uses. That's 30–40% less than building from scratch, almost entirely thanks to MWG + batch_descriptor + adaptation + move kernels being algorithm-agnostic.

---

## Part 6 — Where the architecture does *not* help

Worth being honest about the friction points:

- **The outer NS loop (`run_loop.py:291-540`) is generic in *shape* (adapt → step → terminate → callbacks) but its specific accumulators are NS-flavoured.** It's likely cleaner to add a parallel `run_smc_loop` than to over-parameterize a single shared loop. The callback / termination / adaptation hooks themselves stay; the accumulator vocabulary differs.
- **Step-size adaptation is rate-based.** SMC has different feedback signals — ESS drops, resampling frequency, β step length. The existing rate-based adaptation works *during* the rejuvenation MCMC phase, but you'd also want a temperature/ESS adapter at the outer-loop level. New code, but the `AdaptationManager` framework's hook structure is sane to extend.
- **`PriorMassTermination`** is fundamentally an NS-Skilling construct (`X_i = exp(-i/n_live)` shrinkage). No analog for SMC; entirely replaced by ESS-based criteria.
- **`EnergyLogger` / dead-point trajectory writers** assume "one death per iteration" — convenient for NS but the wrong granularity for SMC. The fix is straightforward (write full snapshots at sane intervals; log per-resampling-event stats) but does involve writing new output code.
- **Multi-GPU sharding (`run_ns_multi_gpu`, `run_ns_sharded`)** is currently NS-flavoured. SMC's global reweight/resample is genuinely harder to shard than NS's per-replica updates — resampling typically needs a global view of weights. Not a blocker (it's a well-studied problem; particle exchange after local resampling is one standard solution), but it's the one place where SMC is *intrinsically* harder than NS to parallelize, irrespective of code quality.

---

## Part 7 — Bottom line

- **Conservative estimate:** ~65% of the existing code is reusable for SMC as-is or with minor tweaks; ~20% needs interface widening; ~15% must be written fresh.
- **Waste-free SMC is barely more work than standard SMC** in this codebase, because the inner scan already produces every intermediate state — the architecture's two-loop structure essentially anticipates it.
- **CLI / config / I/O reuse is high (~70–80%)** — the only fresh I/O work is a weight/ESS logger and an SMC-specific monitor. Most output goes through `write_walker_snapshot` which already exists.
- **Key precondition**: lift per-step `log_likelihood` out of the inner scan (~15 lines). After that, the architecture imposes no further obstacles.
- **The two-loop boundary, batch_descriptor abstraction, and the `MoveKernel` protocol are the three pieces of design foresight that pay off here.** They were built for NS but they are genuinely algorithm-agnostic.

---

## Critical files (for orientation when this becomes implementation)

- `src/jaxrens/sampling/nested_sampling.py:253-445` (`ns_step`) — the scan body to lift `log_likelihood` from, and the template for `smc_step`.
- `src/jaxrens/sampling/run_loop.py:291-540` — the outer-loop skeleton. Inspect for what generalizes; parallel `run_smc_loop` likely.
- `src/jaxrens/sampling/mwg.py` — reuse verbatim.
- `src/jaxrens/sampling/batch_descriptor.py` + `batch_wrapper.py` — reuse verbatim.
- `src/jaxrens/adaptation/manager.py` — reuse verbatim (intra-iteration adaptation); extend at the outer-loop level for temperature/ESS adaptation.
- `src/jaxrens/sampling/termination.py` — widen the protocol; add ESS-based criteria.
- `src/jaxrens/state/ns.py` — template for `SMCState`.
- `src/jaxrens/cli/resolve.py` — where the algorithm dispatch lives.
- `src/jaxrens/cli/parser.py`, `cli/cli.py` — YAML/CLI shell that needs only an `algorithm` field added.
- `src/jaxrens/io/trajectory.py` (`write_walker_snapshot`) — the population-dumping primitive that SMC needs and NS doesn't fully exploit.
- `src/jaxrens/cli/monitor.py` — template for an SMC-specific monitor parallel to `TemperatureCallback`.
- `src/jaxrens/base.py` (`MoveInfo`) — already carries `log_likelihood`; the field exists, it's just not propagated.
