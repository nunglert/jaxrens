# Constraints & alternative coordinate systems — design survey

## Context

This is a **forward-looking architectural assessment**, not an implementation plan. Two related questions:

1. How well does jaxrens accommodate **geometric constraints** on atoms (e.g. fix a surface slab; only adatoms move)? You indicated **per-DOF freezing** (e.g. fix only z of slab atoms) as the desired first step.
2. How well would it accommodate **non-Cartesian coordinates** (e.g. rigid-body / Z-matrix DOFs for molecular crystals)?

The goal is to compare options with complexity estimates and to assess how flexible the current architecture really is for these extensions.

---

## Part 1 — The current architecture, with respect to "what is a walker?"

Relevant load-bearing facts (from `src/jaxrens/state/walker.py`, `sampling/move_kernel.py`, the move kernels, and the backends):

- `WalkerState` is a JAX-registered dataclass with **dynamic** pytree leaves (`positions`, `types`, `energy`, `cell`) and one **static** field (`n_atoms`). The shape contract `*B N 3` is propagated everywhere by the pytree machinery — vmap/pmap "for free". **Adding a new pytree leaf is cheap and idiomatic** (`state/walker.py:78-95`).
- Moves are decoupled from the walker schema via the `MoveKernel` dataclass (`sampling/move_kernel.py`): each move declares its own `extra_state_fields`, which MWG unions into a dynamically-built `MCState`. Galilean already uses this for `direction`. **The schema is genuinely extensible.**
- Cell moves (`volume.py:57-60`, `stretch.py:56-59`, `shear.py:82`) rescale positions uniformly via `einsum`/`transform_positions`. They assume "all atoms ride along with the cell".
- HMC/Galilean use `jax.grad(energy)` for forces (`hmc.py:53,58,69`, `galilean.py:98-118`). Forces are per-atom (N,3) by autodiff structure.
- **Backends compute energy from `(positions, types, cell)` and never return per-atom forces explicitly** (`backends/base.py:37-61`). They are nominally shape-agnostic in the protocol, but each implementation assumes Cartesian (N,3) in its internals (e.g. all-pairs in `lj.py:121-187`; supercell edges in `mace.py:189-222`).
- The NS outer loop (`sampling/nested_sampling.py`) only cares about energies and dead-point replacement. It does **not** make per-atom assumptions and would not need to change for any of the options below.
- Trajectory I/O (`io/trajectory.py`, `io/formats.py`) goes through `ase.Atoms` (extxyz) or dumps `positions/box` (HDF5). Both are Cartesian; rigid-body parameters would need an extra dataset.

The two scarce primitives are: an **atom-subset abstraction** (none exists — no `frozen`, `active`, `mask`, `rigid`, `subset` in the source tree) and a **coordinate parametrization abstraction** (positions are 3N Cartesians, full stop). Everything else is well-factored.

---

## Part 2 — Constraint options (slab-fixing, per-DOF freezing)

The four options below are ordered from cheapest to most invasive. All share the same anchor point: an extra mask field on `WalkerState` (or on `MCState`).

### Option A — Whole-atom freeze via mask in `WalkerState` (~1–2 days)

- Add `frozen_atoms: Bool[Array, "*B N"] | None` as a **dynamic pytree leaf** on `WalkerState` (`state/walker.py`). Vmaps/pmaps for free.
- In each move's `step()`, replace the proposed positions with the old ones for frozen atoms:
  ```python
  new_positions = jnp.where(frozen[..., None], state.positions, new_positions)
  ```
  Touches: `random_walk.py:34-37`, `galilean.py:87-122`, `hmc.py:60-61`, `volume.py:57-60`, `stretch.py:56-59`, `shear.py:82`. Each is a one-line change at the proposal site. `single_atom.py` already operates on one randomly-chosen index — adapt the index sampler to skip frozen atoms.
- For force-driven moves, also zero forces on frozen atoms (`hmc.py:53,69`, `galilean.py:100`) so that momentum/direction does not build up on locked DOFs.
- Cell moves with constraints: pick a policy (see Part 4).
- Config: a new `ConstraintConfig` block on `NSConfig` carrying e.g. `frozen_atom_indices: list[int]` or `frozen_region: {"axis": "z", "below": 5.0}`. Resolver translates this into the mask at init time.

**Complexity**: Small. ~200–400 lines across state, the 7 affected moves, config, and one or two tests. No NS-loop changes.
**Coverage**: Surface slabs, locked spectator atoms, "everything is a partition between frozen and free atoms".
**Limitations**: No per-axis freedom — a slab atom is either fully fixed or fully free.

### Option B — Per-DOF (per-axis) freeze (your stated preference) (~2–4 days)

Same as Option A, but the mask is `Bool[Array, "*B N 3"]` instead of `Bool[Array, "*B N"]`. The `jnp.where` broadcasts trivially since positions are already `(N,3)`.

- Lets you fix `z` of slab atoms while leaving `xy` free (standard in surface DFT).
- HMC/Galilean: zero the corresponding force/momentum components per axis (still a one-line `jnp.where` on the gradient and momentum arrays).
- Cell moves: with per-axis masks, the "lab-frame restore" trick becomes cleaner — restore only the frozen components after the rescale; free components ride the cell.
- DOF counting (Part 5): an effective `n_dof = mask.sum()` is the natural input for any temperature/heat-capacity estimator, and `monitor.py` does not bake in `3 * n_atoms` so adding this is local.

**Complexity**: Still small. Marginal cost over A is mostly correctness review (broadcasting, force masking on the right axis, DOF count plumbing).
**Why this is the recommended starting point**: it strictly subsumes Option A (full freeze = mask all 3 axes) and is no harder to implement.

### Option C — Distance / rigid-bond constraints (SHAKE/RATTLE) (~2–4 weeks)

Constrain selected interatomic distances (bonds, angles via auxiliary distances). Requires iterative constraint solvers that project Cartesian displacements onto the constraint manifold after each proposal. Possible but a different beast:

- Need a constraint Jacobian and an inner Newton/Lagrangian solve, JIT-compiled.
- Acceptance probability gets a Jacobian determinant correction (or use velocity-Verlet+RATTLE for HMC).
- HMC integration becomes constrained leapfrog.

**Complexity**: Big. Probably 1–2 dedicated weeks plus careful testing against a reference.
**When you'd actually want this**: keeping rigid water molecules in solid–liquid coexistence runs, or any time bond lengths must be exactly fixed.
**Verdict**: Defer unless explicitly needed. Option D below subsumes most molecular-crystal use cases without a constraint solver.

### Option D — Active-subset NS (a richer take on A/B)

A generalization worth flagging: instead of a binary mask, store a **list of movable indices** and have moves operate on that slice. Mechanically similar to A/B but with `n_free` as the working size everywhere. This pays off when `n_free << n_atoms` (e.g. one molecule on a large frozen support) because forces, momenta, and proposals can be computed only on the free subset. Same code touchpoints, slightly different style. Probably only worth it if profiling shows masked-but-still-computed work as a bottleneck.

---

## Part 3 — Internal / non-Cartesian coordinates (molecular crystals)

This is qualitatively harder. Three viable paths, ordered cheapest-to-most-ambitious.

### Path I — Rigid-body moves with Cartesian state (recommended) (~1–2 weeks)

Keep `WalkerState.positions` as `(N,3)` Cartesian — backends stay untouched. Add new move kernels that **propose in internal coordinates and reproject to Cartesian** before evaluating the energy.

- New `WalkerState` field (static or dynamic): `molecule_assignment: Int[Array, "*B N"]` mapping each atom to a molecule index, plus per-molecule reference geometry stored in the move's `extra_state_fields`.
- New move: `rigid_body` — proposes `(d_com, d_quat)` per molecule, rebuilds Cartesian positions from `com[m] + R(quat[m]) @ ref_local[a]`. Acceptance uses standard energy difference; for orientation, sample `d_quat` from a small-angle distribution so the proposal is symmetric.
- HMC/Galilean per-molecule: requires defining mass tensors / generalized forces (force on COM = sum of atomic forces; torque on quaternion = sum of `r × f`). Doable in JAX, well-defined math.
- Existing atomic moves (`random_walk`, `single_atom`, …) can coexist for any "still-flexible" atoms.

**Why this works well**: it isolates the coordinate change to *the move kernel itself* — exactly the abstraction the `MoveKernel` protocol is designed for. Backends, the NS loop, the trajectory I/O, and `(N,3)` storage stay as-is. Trajectories are still atomic Cartesians, which is what every downstream tool (ASE, OVITO, VMD) expects.

**Complexity**: Medium. Real work is in the rigid-body integrator (HMC for rotations is non-trivial — quaternion algebra, no-slip dynamics, proper Jacobian) and in seeding `molecule_assignment` + reference geometries from the initial structure. Maybe 1–2 weeks.

### Path II — Hybrid state: keep Cartesians as derived, store generalized coords (~3–6 weeks)

Add a parallel field `gen_coords: Float[Array, "*B Q"]` (e.g. `Q = 6 * n_mol + 3 * n_flex_atoms`), plus a `gen_to_cart(state) -> positions` function in the backend or state utils. Move kernels operate on `gen_coords`; just before energy evaluation, rebuild positions.

- The MoveKernel protocol is permissive enough — `extra_state_fields` already exists. You'd add `gen_coords` as a dynamic leaf and make moves update it instead of (or alongside) `positions`.
- Trajectory I/O would need to either (a) dump Cartesian only (cheap, lossy on the coord parametrization) or (b) extend the writers to dump both.
- Forces in HMC: `dU/dgen = (dU/dCart) @ (dCart/dgen)` — straightforward via `jax.grad` if `gen_to_cart` is JAX-differentiable. Galilean needs the same chain rule for force reflection.
- Risk: any code that reads `state.positions` for non-energy reasons (postprocess, monitors, callbacks) sees stale Cartesians unless they're refreshed each step. Recommend keeping `positions` always current as the canonical mirror and treating `gen_coords` as the "true" DOFs.

**Complexity**: Medium-high. Trickiest part is making sure no part of the codebase silently assumes "positions are the DOFs". A few weeks.

### Path III — Replace Cartesian as the primary representation (~quarter-scale rewrite)

Make `WalkerState.positions` generic — e.g. `dofs: Float[Array, "*B Q"]` with separate metadata describing the parametrization. This is what would be needed if you wanted to support, say, Z-matrix coordinates as the *only* representation.

- `n_atoms` ceases to be the load-bearing shape constant; instead `n_dof` is. Twenty-plus files use `positions.shape[-2]` as `n_atoms` and would need to be audited (`cli/resolve.py`, `cli/cli.py`, all backends, all moves).
- Every backend currently hard-codes Cartesian 3D pairwise/edge math (`lj.py:121-187`, `mace.py:189-222`, `neuralil.py`). They would need an internal-to-Cartesian shim, which is essentially Path I or II anyway.
- Initialization, trajectory I/O, monitors, and any reference to "atom *i*" all need rethinking.

**Complexity**: Very large — easily a quarter of effort. **Not recommended** unless there is a use case that genuinely cannot be expressed as "Cartesian state with internal-coord moves" (Path I) or "hybrid state" (Path II).

---

## Part 4 — Cross-cutting design issues

These are the choices that bite regardless of which option you pick.

### Cell moves with frozen atoms

The interaction between **volume/stretch/shear moves** and **fixed atoms** is genuinely non-trivial, because rescaling the cell rescales fractional coordinates of every atom by definition. Three policies, with no universally right answer:

- **Disable cell moves when constraints are present** — safest. Surface-slab NS is almost always NVT in practice; the slab pins the lab frame.
- **Lab-frame restore** — rescale all positions, then overwrite frozen positions with the originals. Works for small volume changes; can produce ugly geometry (atoms outside the new cell, broken vacuum gap) for large ones.
- **Fractional-frozen** — only the *free* atoms rescale; frozen atoms stay in lab frame; use `N_free` in the `V^N` volume prior at `volume.py:75`. Cleanest physics, most code to touch.

For a first cut, "disable + emit a clear error" is the right default; the right answer can be added later when it has a concrete use case.

### NPT / `V^N` prior

If frozen atoms exist and cell moves remain enabled, the NPT V^N prior (`volume.py:74-76`) and PV term in `EnsembleBackend` (`ensemble.py:76`) should use `N_free`, not `N_total`. Otherwise the prior systematically biases volume away from the physically right scale. Trivial fix once `n_free` is plumbed through.

### Replica exchange

If all replicas share the same constraint mask (the common case), no change needed in `replica_exchange.py`. If masks differ per replica, swap logic gets considerably more delicate — likely better to forbid that combination initially.

### DOF counting for monitors

`monitor.py`'s temperature/heat-capacity estimator does not currently hard-code `3 * n_atoms`, so the change is local: plumb a `n_dof_effective` through the temperature callback. With per-DOF freezing this becomes `mask.sum()` per walker.

### Initialization

Frozen-atom indices fit naturally as metadata on the initial walker file (e.g. an extxyz `fixed` column, or a sidecar JSON). The resolver translates that into the runtime mask. This keeps the *which atoms are fixed* with the structural data and leaves only the *whether constraints are active* in `NSConfig`.

---

## Part 5 — Flexibility assessment

Where the architecture *helps*:

- The pytree-registered `WalkerState` + `MCState` make adding new dynamic fields essentially free for JAX semantics.
- `MoveKernel.extra_state_fields` is exactly the right hook for move-specific extra DOFs (rigid-body refs, masks, momenta). Galilean's `direction` field is the existing precedent.
- The NS outer loop touches positions only as opaque arrays. Adding constraints or alternative coordinates does **not** require any change to `sampling/nested_sampling.py`.
- Backends sit behind a clean protocol (`backends/base.py:37-61`) that takes positions and returns energy. Internal-coord paths can stay invisible to backends if Cartesians are rebuilt before each call.

Where the architecture *fights you*:

- `positions.shape[-2]` is used as `n_atoms` in roughly twenty places (`cli/resolve.py`, `cli/cli.py`, several backends). Anything that decouples `n_dof` from `n_atoms` has to audit all of these.
- Every backend's internals hard-code 3D Cartesian geometry (pair distances, supercell edges, descriptors). They cannot accept generalized coordinates directly.
- Cell moves bake in "all atoms scale with the cell". The constraint case forces a choice between three imperfect policies (see Part 4).
- Trajectory I/O is Cartesian-only. Path-II and Path-III internal coordinates need extended writers.

Overall: **the architecture is well-suited to *additive* extensions (new fields, new moves, new constraints applied at proposal time) and resistant to *replacement* extensions (changing what positions fundamentally are)**. This is good news for both constraint support and rigid-body moves — the natural implementations land cleanly. It is bad news only if you ever want full Z-matrix-only state.

---

## Part 6 — Suggested order, given your stated direction

Given that you picked **per-DOF freezing** and mentioned molecular crystals as a longer-term interest, the path that maximizes leverage is:

1. **Per-DOF freeze (Option B)** as the first concrete feature. Small, useful on its own (surface slabs, mechanical-test fixtures), and the `mask: Bool[*B, N, 3]` field plus the per-move masking pattern set up exactly the plumbing that rigid-body moves will also need (effective DOF count for monitors, force/momentum masking, cell-move policy choice).
2. **Rigid-body moves via Path I** when the molecular-crystal use case becomes concrete. Reuses the constraint infrastructure (rigid-body atoms can be thought of as "atoms whose Cartesian DOFs are slaved to the rigid-body parameters") and adds the rigid-body parametrization purely inside one new move kernel.
3. Revisit cell-move-with-constraints policy and hybrid generalized-coord state (Path II) only if a real workload demands either.

Paths Option C (SHAKE/RATTLE), Option D (active-subset slicing), and Path III (replace Cartesians) are best left until a workload actively requires them.

---

## Critical files (for reference if/when this becomes implementation)

- `src/jaxrens/state/walker.py` — where the mask field would live.
- `src/jaxrens/sampling/move_kernel.py` & `sampling/mwg.py` — how to declare per-move extra state and thread the mask through.
- `src/jaxrens/sampling/moves/{random_walk,galilean,hmc,single_atom,volume,stretch,shear}.py` — the seven proposal sites that need masking.
- `src/jaxrens/backends/ensemble.py` (`:76`) and `sampling/moves/volume.py` (`:74-76`) — `N → N_free` plumbing for the V^N prior.
- `src/jaxrens/state/config.py` — `ConstraintConfig` would attach here (mirroring `InterREConfig`).
- `src/jaxrens/init/structure.py`, `init/positions.py` — where atom-level metadata enters the run; natural home for frozen-atom indices.
