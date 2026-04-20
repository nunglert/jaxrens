# Pathology: slow Monitor observables / plot.py takes minutes

## Symptom

`plot.py` (or any script calling `Monitor.heat_capacity(T)` / `Monitor.log_Z(T)` with a large T array) runs for minutes per figure. Expected: seconds.

## Root causes (cumulative — both fixed 2026-04-18)

### A. Python for-loop over betas in Monitor methods

Original pattern in `postprocess/monitor.py`:
```python
results = np.array([
    float(_heat_capacity(float(b), dead_e, live_e, ...))
    for b in betas
])
```
Each iteration invokes a JAX function WITHOUT `@jax.jit`. JAX dispatch cost ~1ms per call, 200 betas × 5 observables ≈ 1s. With GPU initialization per call, can balloon to minutes.

Fix: `jax.vmap + jax.jit` once at observable-method call, evaluated across the beta axis:
```python
jitted = jax.jit(scalar_fn)
vmapped = jax.vmap(lambda b: jitted(b, **fn_kwargs), in_axes=0)
return np.asarray(vmapped(jnp.asarray(betas)))
```
One trace, vectorized over all betas. 100×+ speedup.

### B. Nested logsumexp-on-growing-slice loop in plot_log_evidence_trace

Original:
```python
for k in range(1, n_dead + 1):
    cumulative[k-1] = float(logsumexp(log_w[:k] + log_L[:k]))
```
Each `k` creates a slice of distinct shape → JAX retrace per k → n_dead compilations.

Fix: pure-numpy running accumulator:
```python
running = -np.inf
for i in range(n_dead):
    running = np.logaddexp(running, log_terms[i])
    cumulative[i] = running
```
O(n_dead) total. No JAX.

## Detection heuristic

If plot.py > 30s and `Monitor` was recently touched, suspect (A). Time individual observable calls:
```python
t0 = time.monotonic(); mon.log_Z(T); print(time.monotonic()-t0)
```
Healthy: <0.5s first call, <0.15s cached. Unhealthy: >5s per call.

If (A) is clean but `plot_log_evidence_trace` is slow in isolation, suspect (B).

## Future guard

Any new `Monitor.<observable>(T)` method must:
- Vmap over T internally via `_vmap_over_beta` (or equivalent)
- Not build fresh `jax.jit` wrappers per call
- Avoid Python loops over T
