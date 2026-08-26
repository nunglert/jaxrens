---
myst:
  html_meta:
    "description": "Symbols and notation reference for jaxrens"
---

# Notation and symbols

This page is the single source of truth for what each symbol in
the jaxrens documentation means. The same letter is sometimes
overloaded across domains (a *pressure* `P` and a *replicas-per-GPU*
`P` live in different paragraphs but on the same page); when in
doubt, look it up here. New documentation should reference symbols
defined here rather than introducing fresh ones.

## Code-side shape conventions

The whole NS state is shaped `(G, P, K, …)` under the most general
descriptor, with leading axes peeled off for simpler runs.

| Symbol | Meaning | Where it lives |
|---|---|---|
| $G$ | `n_gpu` — number of devices in the pmap axis. `G = 1` for single-GPU runs. | leading axis under {class}`~jaxrens.sampling.batch_descriptor.PmapVmapRuns` |
| $P$ | `n_per_gpu` — number of NS instances per device. **Not pressure** in this section. | second axis under `PmapVmapRuns` |
| $R$ | `n_runs` — total parallel NS instances. Equal to $G \cdot P$ when both axes are flattened. | leading axis under {class}`~jaxrens.sampling.batch_descriptor.VmapRuns` |
| $K$ | `n_walkers` — live walkers per NS instance. | walker batch axis on every state field |
| $N_\mathrm{atoms}$ | `n_atoms` — atoms per configuration. Static field; changing it triggers a JIT recompile. | trailing axis on `positions`, `types`, etc. |
| $S$ | `n_species` — distinct species labels. Static field. | length of $\mathbf c_i$, $\boldsymbol\mu_i$, $\mathbf N(\sigma)$ |
| $M$ | number of move kernels in the MWG composer | length of `step_sizes` |

A typical positions array under {class}`~jaxrens.sampling.batch_descriptor.PmapVmapRuns`
has shape $(G, P, K, A, 3)$; under `VmapRuns` it collapses to
$(R, K, A, 3)$; under {class}`~jaxrens.sampling.batch_descriptor.SingleRun`
to $(K, A, 3)$.

## Nested-sampling quantities

| Symbol | Meaning |
|---|---|
| $r$ | A single configuration in parameter space. In atomistic NS this bundles positions $\mathbf q$, cell $h$, and species $\sigma$. |
| $\pi(r)$ | Prior density on configurations. |
| $L(r)$ | Likelihood. In jaxrens we work with $L = e^{-H}$, where $H$ is the Hamiltonian (energy-like). |
| $H(r)$ | Hamiltonian / target. For NPT, $H = U + P V$; for NVT, $H = U$; for semi-grand, $H = U - \boldsymbol\mu \cdot \mathbf N$. |
| $U(r)$ | Bare potential energy (no $PV$ or $-\mu N$ term). What backends return. |
| $Z$ | Evidence — the integral $\int L(r)\,\pi(r)\,\mathrm d r$. NS estimates this. |
| $X$ | Prior mass in the constrained-likelihood transform: $X(L^\star) = \int_{L > L^\star} \pi\,\mathrm d r$. Maps the multi-D integral onto $X \in [0, 1]$. |
| $L_i^\star$ | Likelihood threshold at iteration $i$. Equals the *worst* live walker's likelihood when the walker was retired. |
| $E_{\max}$ | Hamiltonian threshold equivalent to $L_i^\star$. The constraint MCMC enforces is $H(r) < E_{\max}$. |
| $X_i$ | Prior mass remaining at iteration $i$. Expectation $X_i \approx \exp(-i / N)$ for $K = N$ live walkers and $n_\text{cull} = 1$. |
| $w_i$ | Trapezoid quadrature weight: $w_i = \tfrac{1}{2}(X_{i-1} - X_{i+1})$. |
| $i$ | Outer-loop iteration index (Python). Used in `manager.fires(i)` and similar. |

## Thermodynamic ensemble parameters

These are per-replica scalars or vectors stored in
`MCState.ensemble_params`. In multi-replica runs each is indexed by
the replica $i \in \{0, \dots, n_\text{total} - 1\}$.

| Symbol | Meaning |
|---|---|
| $P$ | Pressure (NPT). When ambiguous with the pmap-axis `P`, this section uses $P_i$ explicitly. |
| $V$ | Cell volume, $V(h) = \lvert \det h \rvert$ for cell matrix $h$. |
| $T$ | Temperature. **Not a free parameter in NS** — the role of temperature is played by the threshold $E_{\max}$. Mentioned only in cross-references to ordinary MCMC. |
| $\boldsymbol\mu$ | Per-species chemical potential vector, length $S$. |
| $\mathbf N(\sigma)$ | Species-count vector of a configuration, $\mathbf N(\sigma)_s = \#\{a : \sigma_a = s\}$. |
| $\mathbf c$ | Target composition vector, $\sum_s c_s = A$. Each XRENS replica fixes one. |
| $\Omega$ | Grand-canonical energy used in semi-grand swaps: $\Omega = U - \boldsymbol\mu \cdot \mathbf N(\sigma)$. |
| $h$ | Cell matrix (3×3), columns are the lattice vectors. |
| $\mathbf q$ | Position array (per atom: ℝ³, per walker: $(A, 3)$). |
| $\sigma$ | Species / type array. Integer-valued, per-atom. |

## MWG and move kernels

| Symbol | Meaning |
|---|---|
| $k$ | Move-kernel index, $k \in \{0, \dots, M-1\}$. Stored in `MoveInfo.move_idx`. |
| $w_k$ | Move weight (relative). Set in YAML or `MoveConfig`. |
| $p_k$ | Move probability $p_k = w_k / \sum_j w_j$. |
| $\alpha(r \to r')$ | Acceptance for an MCMC proposal from $r$ to $r'$. Under NS it reduces to $\mathbf 1[H(r') < E_{\max}]$. |
| $n_\text{mcmc}$ | `n_mcmc_steps` — MCMC steps per walker per outer iteration. |
| $n_\text{cull}$ | Walkers retired per iteration (default 1). Affects $X_i$ contraction. |
| $n_\text{extra}$ | Extra random live walkers reseeding the MCMC chain per iteration. |
| $\sigma_k$ | Step size of move $k$. Stored per-walker in `MCState.step_sizes` so adaptation can be vectorised. |

The four reject-reason codes used by the trial-rate counters are
constants: `0 = accepted`, `1 = energy`, `2 = cell`, `3 = prior`.

## Step-size adaptation

| Symbol | Meaning |
|---|---|
| `min_rate`, `max_rate` | Target acceptance window, e.g. $[0.3, 0.5]$. |
| `adjust_factor` | Multiplicative bisection factor (>1). |
| `adjust_n_samples` | Walkers sampled per bisection round. |
| `adjust_max_rounds` | Cap on `lax.while_loop` iterations. |
| `adjust_interval` | Outer-loop iterations between adaptation calls; `0` disables. |

## Replica exchange

Replicas share state shape; what differs is `ensemble_params`.

| Symbol | Meaning |
|---|---|
| $i, j$ | Replica indices. Edge labels in inter-RE diagrams use $(i, j)$ for an attempted swap between replica $i$ and $j$. |
| `every` | Outer-loop iterations between swap passes. |
| `n_swap_cycles` | Number of (even, odd) phases per fire. |
| $\alpha_{ij}$ | Swap acceptance. Form depends on flavor — see {doc}`/user/concepts/replicas` for the three derived expressions (Pressure-RENS, XRENS, semi-grand). |
| $\mathcal{M}_{\mathbf c}(\sigma, \xi)$ | Random morph operator: re-permutes species labels of $\sigma$ to a new vector with composition exactly $\mathbf c$. $\xi$ is the auxiliary randomness. |
| $\tilde U_i$ | Re-evaluated potential after a morph: $\tilde U_i = U(\mathbf q_j, \mathcal{M}_{\mathbf c_i}(\sigma_j), h_j)$. Two such evaluations per XRENS swap pair. |
