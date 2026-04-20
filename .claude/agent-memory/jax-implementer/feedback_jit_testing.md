---
name: JIT testing policy
description: All JIT-compatible functions must be tested under jax.jit
type: feedback
---

Every function that is JIT-compatible must have at least one test that calls it under `jax.jit`. This is a non-negotiable project policy to catch tracing issues at test time rather than at run time.

**Why:** Silent tracing bugs (Python control flow on traced values, dynamic shapes) only surface when code runs under JIT. Tests that only call functions eagerly give false confidence.

**How to apply:** For every new public function added, check whether it is JIT-compatible. If yes, add a `jax.jit(fn)(...)` call in the test suite.
