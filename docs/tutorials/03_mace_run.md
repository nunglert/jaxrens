# MACE: 16-atom silicon, NPT

Same shape as {doc}`02_lj_run`, with a machine-learned potential instead of
Lennard-Jones. Only the `backend:` block really changes — which is the point:
swapping the energy model does not change how you drive the code.

Expect this one to take meaningfully longer. MACE evaluates a message-passing
network per walker, and the first iterations pay JIT compilation for a much
larger graph than LJ.

## 0. Get a JAX-loadable model

MACE checkpoints are distributed as torch weights; converting is a one-off
shell step:

```bash
# certifi CA bundle — only needed when downloading a foundation model
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')

mace-jax-from-torch --foundation mp --model-name small --output small-jax.npz
```

That writes `small-jax.npz` (msgpack params, despite the extension) plus a
sibling `small-jax.json`. Point `checkpoint_path` at the `.npz`; the loader
finds the `.json` beside it. A bundle directory, a `.json`/`.msgpack` pair, an
orbax `.ckpt` and a `.pkl` all work too — one loader handles all five.

The config below points at the tiny bundled fixture in `tests/_assets/` so it
runs from a fresh checkout with no download. For real work, point it at the
model you just converted. See {doc}`/user/mace_models` for the details.

## The config

`examples/tutorials/03_mace_si16/config.yaml`:

```{literalinclude} ../../examples/tutorials/03_mace_si16/config.yaml
:language: yaml
```

Three MACE-specific concerns, none of which existed in the LJ tutorial:

**`supercell_trafo` follows the model's cutoff.** MACE-MP has
`r_cutoff = 6 Å`, so the tiled cell must span at least 12 Å. Too small and the
model silently sees a truncated environment.

**`max_neighbors_list` is a recompilation budget.** The neighbour buffer is a
compile-time shape. When a move pushes atoms closer than the current rung can
hold, the loop escalates to the next one and JAX compiles a fresh kernel. A
short ladder bounds how many times that can happen; `max_neighbors_offset`
adds headroom so a marginal increase does not immediately re-trip it.

**Cell moves carry most of the weight.** At fixed composition, the degrees of
freedom that matter for a crystal are volume and shape, so `volume`, `shear`
and `stretch` get 32 of the 34 total weight between them.

## 1. Validate — and use `--full` here

```bash
cd examples/tutorials/03_mace_si16
jaxrens validate -c config.yaml --full
```

The `--full` tier matters much more with an MLIP than it did with LJ: it is
what actually loads the model file and evaluates an initial energy. The
default tier confirms the path exists, not that the checkpoint is loadable or
that its z-table covers silicon. Paying tens of seconds here beats discovering
it after a job has queued.

## 2. Run

```bash
jaxrens run -c config.yaml
tail -f si16.log        # from a second shell; `run` prints nothing
```

Two things in the log are worth watching that LJ never showed you.

**Bucket escalation.** Lines about the neighbour bucket being raised are
normal a few times early on, as the walkers relax into a sensible density.
They are a problem only if they never stop — that means `max_neighbors_list`
has no comfortable rung, and every escalation is another compile. Turn on
`output.save_max_neighbors` to record the observed counts and plot them.

**Step sizes collapsing.** If `Emax` stalls while the adaptation trace drives
step sizes toward zero, walkers have likely fallen into a close-contact region
where the model is unphysical. Foundation models without a proper short-range
repulsive term do this. The fix is `backend.softcore_repulsion`, or a
`minimum_distance` constraint — {doc}`/user/troubleshooting` covers both, and
the difference between them.

## 3. Inspect

```bash
ls output/
jaxrens plot output/si16.energies
jaxrens plot output/si16.adaptation.h5
```

Same artifacts as the LJ run. With `output.save_max_neighbors: true` you also
get `si16.max_neighbors.h5`, and `jaxrens plot` renders it as per-walker
neighbour-count percentiles with the bucket ladder overlaid — the fastest way
to tell whether your ladder is sized right.

## 4. Scaling up

This config is a smoke test, not a production run: 32 walkers and 200
iterations will not resolve the silicon phase diagram. For real work raise
`run.n_live` into the hundreds and let `termination.prior_mass` decide when to
stop rather than a fixed iteration count.

At that point one GPU stops being enough, and a list-valued `ensemble.pressure`
fans the run across replicas with pressure-RENS swaps between them — see
{doc}`/user/concepts/replicas` for how the topology is derived and
{doc}`/reference/config` for the `inter_re:` surface.
