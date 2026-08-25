# A point in a landscape

The smallest thing nested sampling can be asked to do, and the best place to
see what it *is* before any materials complexity arrives. One particle, a
fixed landscape of Gaussian wells, no periodic cell, no pressure, no cell
moves. It runs in about five seconds on a CPU.

## The problem

The energy is a Gaussian mixture — a sum of wells in the xy-plane:

$$
E(\mathbf{r}) = -\log \sum_k \exp\!\left(
  -\frac{\lVert \mathbf{r} - \boldsymbol{\mu}_k \rVert^2}{2\sigma^2}
\right)
$$

Three of the wells sit far apart and are degenerate. The other two are placed
close enough that their Gaussians overlap and merge into a **single deeper
basin** — so there is one unique global minimum, at `E ≈ -0.92` against
`E ≈ -0.39` for the isolated wells.

That asymmetry is the whole exercise. Nested sampling has to find the deeper
basin *without being told where it is*, by shrinking a contour of constant
energy until only that basin survives inside it.

```{image} ../_static/figures/tutorials/gauss2d_surface.png
:alt: the Gaussian-mixture energy surface, and the walker population contracting onto the deepest basin
:width: 100%
```

Left: the surface itself, with the five Gaussian centres marked — note the two
at the top right sitting almost on top of each other, which is what makes that
basin deeper than the rest. Right: the live population, dumped every 500
iterations by `output.snapshot_interval`. At iteration 500 the walkers are
spread across all four basins; by 1500 each basin holds a tight cluster; by
2500 only the merged one is still occupied. That contraction *is* the
algorithm — everything the log prints below is a number describing this
picture.

## The config

`examples/tutorials/00_gaussian_2d/config.yaml`:

```{literalinclude} ../../examples/tutorials/00_gaussian_2d/config.yaml
:language: yaml
```

Notice how much is *absent*. There is one move, because with a single particle
and no cell there is nothing else to move — no `volume`, `shear` or `stretch`,
since there is no cell to act on. `ensemble: nvt` because there is no volume
for a `P·V` term to multiply. `backend.periodic: false`, so no images and no
cutoff. The `cell:` section is present only to bound where the initial
position is drawn from; it is never sampled.

## Run it

```bash
cd examples/tutorials/00_gaussian_2d
jaxrens validate -c config.yaml --full
jaxrens run -c config.yaml
tail -f gauss2d.log     # from a second shell; `run` prints nothing
```

```text
Starting NS run: 200 walkers, 1 atoms, max_iter=3000, n_mcmc=20
iter=0     Emax=85.4479    log_Z=-90.7462  dt=1.7s  nE=7.90e+02
iter=250   Emax=18.3548    log_Z=-22.5746  dt=0.6s  nE=5.93e+03
iter=500   Emax=6.76157    log_Z=-10.8761  dt=0.2s  nE=1.12e+04
iter=1000  Emax=0.852648   log_Z=-5.5240   dt=0.2s  nE=2.17e+04
iter=2000  Emax=-0.870357  log_Z=-4.5496   dt=0.3s  nE=4.78e+04
iter=2750  Emax=-0.913359  log_Z=-4.5468   dt=0.3s  nE=5.82e+04
NS finished: 3000 iterations, log_Z=-4.5466, elapsed=5s
```

This is the clearest place to read the two columns that matter.

`Emax` is the energy contour. It starts at 85 — walkers scattered anywhere —
and falls monotonically to `-0.913`, just above the true minimum of `-0.9225`.
It **cannot** go back up: that is the defining constraint of the algorithm.

`log_Z` is the accumulated log-evidence, and it does the opposite. It climbs
steeply while the contour is still sweeping up huge volumes of configuration
space, then flattens — by iteration 2000 it has moved less than 0.01. That
flattening is convergence: the remaining prior mass is too small to contribute
regardless of how deep the energy goes. It is what `termination.prior_mass`
detects.

The final 200 dead points sit at `x = 1.80 ± 0.03`, `y = 1.60 ± 0.02` — the
bottom of the merged basin. NS found it among five wells without gradients and
without being told it existed.

## The plots

```bash
jaxrens plot output/gauss2d.energies
jaxrens plot output/gauss2d.adaptation.h5
```

```{image} ../_static/figures/tutorials/gauss2d.energies.png
:alt: dead-point energy trail for the 2-D Gaussian mixture
:width: 100%
```

The energy trail is the ladder of dead points — every walker that was culled,
in order. The staircase is NS working: long flat stretches where the contour
grinds through a wide, shallow region, then drops as it clears a barrier into
something deeper.

```{image} ../_static/figures/tutorials/gauss2d.adaptation.png
:alt: step-size and acceptance-rate traces
:width: 100%
```

The adaptation plot is the diagnostic you will reach for most often. The step
size shrinks as the accessible region does — that is correct and expected.
What you are checking is the acceptance rate: it should sit inside the band
set by `adaptation.defaults.min_rate`/`max_rate`. Pinned at zero means the
move is too large to ever be accepted; pinned at one means it is so small the
walker is not going anywhere.

## Next

- {doc}`01_rens_toy` — the same machinery with replicas and exchange, still
  small enough to see everything.
- {doc}`/user/concepts/ns_loop` — the mathematics behind the two columns above.
