---
name: pathology-pressure-rens-double-pv
description: Pressure-RENS swap acceptance decays to 0 over ~100 iters because PressureRENSSwap.accept treats stored enthalpy as raw U, double-counting P·V
metadata:
  type: project
---

## Symptom

`inter_re acc` rate for `flavor: pressure` RENS decays smoothly across NS iterations: ~0.15 at iter=4, ~0.03 at iter=48, 0.00 by iter ~110. Same setup runs fine in legacy `jaxnest_dev`. Affects any NPT pressure-stagger RENS run; severity scales with the pressure spread × volume × Emax tightness.

## Root cause

`PressureRENSSwap.accept` in `src/jaxrens/sampling/moves/replica_exchange.py:190-214` (current production branch, commit ~1f6d9d6) assumes `proposed["energy_a"] = U_A` (raw potential) and computes:

```python
h_a_in_b = e_a + p_b * v_a   # intended as U_A + P_B*V_A
```

But the stored `pop.energy` field in NPT mode is **already** enthalpy at the run's own pressure: `EnsembleBackend.__call__` (`src/jaxrens/backends/ensemble.py:76`) returns `H = U + P_self * V`. So jaxrens actually evaluates:

```
(U_A + P_A*V_A) + P_B*V_A = U_A + (P_A + P_B)*V_A      ← +P_A*V_A too much
```

Legacy `create_perform_pressure_swap` in `jaxnest_dev/src/jaxnest/replica_exchange.py:144-189` correctly subtracts the old PV first:

```python
pv_old   = eval_pv_batch(pair_pressure, cell)      # P_self * V
e_pot    = stored_energy - pv_old                  # recovers U
pv_new   = eval_pv_batch(pair_pressure[::-1], cell)# P_partner * V
energy_new = (e_pot + pv_new)[::-1]                # U + P_partner*V
```

Diagnostic `/home/nico.unglert/dump/probe_pressure_rens.py` reproduces the exact decay: at Si16, P_A=1 GPa, P_B=2 GPa, V~320 Å³, the spurious +P·V term adds 2–4 eV to each leg of the test, which exceeds the typical Emax gap once NS has run ~50 iterations.

In the 100k-sample sweep at representative iter=24 values: legacy 78% accept → jaxrens 5% accept (15× collapse).

## Canonical diagnosis

1. Confirm `flavor: pressure` and `backend.ensemble.type: npt` (i.e. EnsembleBackend wraps the base, so stored energy is enthalpy).
2. Read `accept()` in `replica_exchange.py` and verify it does NOT subtract `P_self * V` from the stored energy before adding `P_partner * V`.
3. Plot `inter_re acc` vs iter — exponential-ish decay synchronized with Emax tightening is the signature. Constant-Emax-gap regimes (early iters) leak slower; tight regimes collapse.

## Fix recipe

In `PressureRENSSwap.accept` (`src/jaxrens/sampling/moves/replica_exchange.py:203-212`), recover U before re-adding PV:

```python
if use_pressure:
    v_a = _get_volume(proposed["cell_a"])
    v_b = _get_volume(proposed["cell_b"])
    # Stored energies are enthalpies at each run's own pressure (EnsembleBackend
    # adds P_self * V).  Subtract the old PV to recover the raw potential U,
    # then re-add PV at the partner's pressure.
    u_a = e_a - p_a * v_a
    u_b = e_b - p_b * v_b
    h_a_in_b = u_a + p_b * v_a
    h_b_in_a = u_b + p_a * v_b
    accepted = (h_a_in_b < emax_b) & (h_b_in_a < emax_a)
```

Note: the `perform_swap` back-compat shim at line 252 documents `energies_pair` as "energies" generically — its semantics matched the original buggy accept(). Existing unit tests in `tests/test_replica_exchange.py` likely encoded the wrong convention. After the fix, audit tests for the same enthalpy-vs-U assumption.

## Detection heuristic

Any time `inter_re` for `flavor: pressure` shows a monotone acceptance decay over iterations (not a noise floor — a smooth decay), check this acceptance formula first. The legacy code's `pv_old`/`pv_new` two-step is the canonical recipe.

## Related

- [[lesson_initial_step_size_plumbing]] — similar "stored value semantics mismatch" pattern
- jaxnest_dev legacy reference: `src/jaxnest/replica_exchange.py:115-191`, `src/jaxnest/backends/general.py:65-72`
