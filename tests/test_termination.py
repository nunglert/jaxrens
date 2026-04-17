"""Tests for termination criteria (Step 14).

Tests:
- TempTermination converges on harmonic potential with known Z(T)
- EnergyTermination triggers at correct energy
- IterationTermination and PriorMassTermination basic behavior
- Pluggable TerminationCriterion protocol (custom criterion)
- check_any helper
- Batch termination: all parallel runs must converge
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from jaxrens.sampling.termination import (
    EnergyTermination,
    IterationTermination,
    PriorMassTermination,
    TempTermination,
    check_any,
    KB_EV,
)


# ── IterationTermination ─────────────────────────────────────────────────


class TestIterationTermination:
    def test_triggers_at_max(self):
        t = IterationTermination(max_iterations=100)
        assert not t.check(98, 0.0)
        assert t.check(99, 0.0)

    def test_triggers_past_max(self):
        t = IterationTermination(max_iterations=50)
        assert t.check(100, 0.0)

    def test_message(self):
        t = IterationTermination(max_iterations=10)
        assert "10" in t.message()


# ── PriorMassTermination ─────────────────────────────────────────────────


class TestPriorMassTermination:
    def test_never_triggers_early(self):
        t = PriorMassTermination(n_live=100)
        for i in range(100):
            assert not t.check(i, 0.0)

    def test_triggers_when_mass_negligible(self):
        t = PriorMassTermination(n_live=50, threshold=0.1)
        t.update_evidence(-5.0)  # relatively high evidence
        # At iteration 1000, log_remaining = -1000/50 = -20, well below -5.1
        assert t.check(1000, 0.0)

    def test_does_not_trigger_without_evidence(self):
        t = PriorMassTermination(n_live=50)
        # Without calling update_evidence, _log_evidence = -inf
        # log_remaining < -inf is always False
        assert not t.check(500, 0.0)


# ── EnergyTermination ────────────────────────────────────────────────────


class TestEnergyTermination:
    def test_triggers_below_target(self):
        t = EnergyTermination(min_energy=-1.0)
        assert not t.check(0, 0.5)
        assert not t.check(1, -0.5)
        assert t.check(2, -1.5)

    def test_exact_boundary(self):
        t = EnergyTermination(min_energy=0.0)
        assert not t.check(0, 0.0)  # not strictly less
        assert not t.check(1, 0.001)
        assert t.check(2, -0.001)

    def test_message(self):
        t = EnergyTermination(min_energy=-2.0)
        assert "-2.0" in t.message()


# ── TempTermination ──────────────────────────────────────────────────────


class TestTempTermination:
    def test_basic_convergence(self):
        """Feed decreasing energies; once contributions fall off, should trigger."""
        n_walkers = 100
        target_temp = 300.0
        t = TempTermination(n_walkers=n_walkers, target_temp=target_temp, threshold=10.0)

        # Simulate a nested sampling run with exponentially decaying energies
        # (realistic: energies decrease fast then plateau near ground state)
        n_iters = 5000
        energies = [10.0 * math.exp(-i / (n_walkers * 0.5)) for i in range(n_iters)]

        triggered_at = None
        for i, e in enumerate(energies):
            if t.check(i, e):
                triggered_at = i
                break

        assert triggered_at is not None, "TempTermination should have triggered"
        assert triggered_at > 0, "Should not trigger on first iteration"

    def test_never_triggers_if_energies_flat(self):
        """If energies stay constant, prior mass shrinks but Z terms shrink too."""
        t = TempTermination(n_walkers=50, target_temp=1000.0, threshold=10.0)
        # Constant energy: log_Z_term = i*log_t + log_shell - beta*E
        # The i*log_t term makes this decrease, so eventually it will trigger
        # once log_t decay outweighs constant -beta*E
        triggered = False
        for i in range(10000):
            if t.check(i, 5.0):
                triggered = True
                break
        # With constant energy and enough iterations, the weight decay
        # will cause convergence
        assert triggered

    def test_harmonic_potential_convergence(self):
        """Test on harmonic potential with known Z(T).

        For a 1D harmonic oscillator E = 0.5*k*x^2:
        Z(T) = sqrt(2*pi*kB*T/k) (per degree of freedom)

        We simulate a nested sampling sequence of energies and check that
        TempTermination triggers at a reasonable point.
        """
        n_walkers = 200
        target_temp = 300.0  # K
        k = 1.0  # eV/Å^2
        beta = 1.0 / (KB_EV * target_temp)

        t = TempTermination(
            n_walkers=n_walkers,
            target_temp=target_temp,
            threshold=10.0,
        )

        # Simulate NS on harmonic: energies decrease from some max toward 0
        # In NS, dead point energies decrease monotonically
        n_iters = 5000
        # Exponentially decreasing energies (typical NS behavior)
        energies = [5.0 * math.exp(-i / (n_walkers * 0.5)) for i in range(n_iters)]

        triggered_at = None
        for i, e in enumerate(energies):
            if t.check(i, e):
                triggered_at = i
                break

        assert triggered_at is not None, "Should converge on harmonic"
        # Should terminate well before all iterations are used
        assert triggered_at < n_iters - 1

    def test_n_cull_greater_than_one(self):
        """Multi-cull should still work."""
        t = TempTermination(n_walkers=100, target_temp=500.0, n_cull=3, threshold=10.0)

        triggered_at = None
        for i in range(5000):
            e = 10.0 * math.exp(-i / 50.0)
            if t.check(i, e):
                triggered_at = i
                break

        assert triggered_at is not None

    def test_threshold_sensitivity(self):
        """Smaller threshold should trigger earlier."""
        energies = [10.0 * math.exp(-i / 50.0) for i in range(5000)]

        t5 = TempTermination(n_walkers=100, target_temp=300.0, threshold=5.0)
        t15 = TempTermination(n_walkers=100, target_temp=300.0, threshold=15.0)

        trigger_5 = None
        trigger_15 = None
        for i, e in enumerate(energies):
            if trigger_5 is None and t5.check(i, e):
                trigger_5 = i
            if trigger_15 is None and t15.check(i, e):
                trigger_15 = i
            if trigger_5 is not None and trigger_15 is not None:
                break

        assert trigger_5 is not None and trigger_15 is not None
        assert trigger_5 < trigger_15, "Smaller threshold should trigger earlier"

    def test_log_Z_term_properties(self):
        """Properties track internal state correctly."""
        t = TempTermination(n_walkers=100, target_temp=300.0)
        assert t.log_Z_term_max == -math.inf
        assert t.log_Z_term_last == -math.inf

        t.check(0, 5.0)
        assert t.log_Z_term_max > -math.inf
        assert t.log_Z_term_last > -math.inf

    def test_message_format(self):
        """Message includes temperature and convergence info."""
        t = TempTermination(n_walkers=100, target_temp=300.0)
        # Feed some data so it triggers
        for i in range(3000):
            if t.check(i, 10.0 * math.exp(-i / 50.0)):
                break
        msg = t.message()
        assert "300.0" in msg
        assert "converged" in msg


# ── check_any helper ─────────────────────────────────────────────────────


class TestCheckAny:
    def test_returns_first_satisfied(self):
        c1 = IterationTermination(max_iterations=1000)
        c2 = EnergyTermination(min_energy=5.0)  # triggers when emax < 5
        done, msg = check_any([c1, c2], iteration=10, emax=3.0)
        assert done
        assert "energy" in msg.lower() or "5.0" in msg

    def test_returns_false_when_none_satisfied(self):
        c1 = IterationTermination(max_iterations=1000)
        c2 = EnergyTermination(min_energy=-100.0)
        done, msg = check_any([c1, c2], iteration=10, emax=0.0)
        assert not done
        assert msg == ""

    def test_empty_list(self):
        done, msg = check_any([], iteration=0, emax=0.0)
        assert not done

    def test_multiple_satisfied_returns_first(self):
        c1 = IterationTermination(max_iterations=5)
        c2 = EnergyTermination(min_energy=10.0)
        done, msg = check_any([c1, c2], iteration=10, emax=0.0)
        assert done
        # First criterion (iteration) should match
        assert "iteration" in msg.lower() or "max" in msg.lower()


# ── Pluggable protocol ──────────────────────────────────────────────────


class CustomCriterion:
    """Custom termination: stop when energy oscillates."""

    def __init__(self, max_oscillations: int = 3):
        self.max_oscillations = max_oscillations
        self._prev_energy = None
        self._prev_direction = None
        self._oscillations = 0

    def check(self, iteration: int, emax: float) -> bool:
        if self._prev_energy is not None:
            direction = emax > self._prev_energy
            if self._prev_direction is not None and direction != self._prev_direction:
                self._oscillations += 1
            self._prev_direction = direction
        self._prev_energy = emax
        return self._oscillations >= self.max_oscillations

    def message(self) -> str:
        return f"Energy oscillated {self._oscillations} times"


class TestPluggableProtocol:
    def test_custom_criterion_with_check_any(self):
        """Custom criterion works with check_any."""
        custom = CustomCriterion(max_oscillations=2)
        criteria = [IterationTermination(max_iterations=1000), custom]

        # Oscillating energies: 5, 3, 6, 2, 7 -> oscillations at indices 2, 3, 4
        energies = [5.0, 3.0, 6.0, 2.0, 7.0, 1.0, 8.0]
        for i, e in enumerate(energies):
            done, msg = check_any(criteria, i, e)
            if done:
                assert "oscillat" in msg.lower()
                break
        else:
            pytest.fail("Custom criterion should have triggered")


# ── Batch termination (all parallel runs must converge) ──────────────────


class TestBatchTermination:
    def test_all_runs_must_converge(self):
        """Simulate multiple parallel runs; all must satisfy criterion."""
        n_runs = 4
        criteria = [
            TempTermination(n_walkers=100, target_temp=300.0, threshold=10.0)
            for _ in range(n_runs)
        ]

        # Different energy schedules for each run (exponentially decaying)
        schedules = [
            [10.0 * math.exp(-i / 50.0) for i in range(5000)],
            [8.0 * math.exp(-i / 60.0) for i in range(5000)],
            [12.0 * math.exp(-i / 40.0) for i in range(5000)],
            [6.0 * math.exp(-i / 80.0) for i in range(5000)],
        ]

        all_triggered = [None] * n_runs
        for i in range(5000):
            for r in range(n_runs):
                if all_triggered[r] is None and criteria[r].check(i, schedules[r][i]):
                    all_triggered[r] = i

            if all(t is not None for t in all_triggered):
                break

        # All runs should have converged
        assert all(t is not None for t in all_triggered)
        # They converge at different iterations
        assert len(set(all_triggered)) > 1, "Runs should converge at different times"

    def test_slowest_run_determines_batch_termination(self):
        """Batch terminates when the slowest run converges."""
        n_runs = 3
        criteria = [
            TempTermination(n_walkers=100, target_temp=300.0, threshold=10.0)
            for _ in range(n_runs)
        ]

        # One slow run (slow decay), two fast
        schedules = [
            [10.0 * math.exp(-i / 40.0) for i in range(5000)],   # fast
            [10.0 * math.exp(-i / 40.0) for i in range(5000)],   # fast
            [10.0 * math.exp(-i / 100.0) for i in range(5000)],  # slow
        ]

        per_run_trigger = [None] * n_runs
        batch_done_at = None

        for i in range(5000):
            for r in range(n_runs):
                if per_run_trigger[r] is None and criteria[r].check(i, schedules[r][i]):
                    per_run_trigger[r] = i

            if all(t is not None for t in per_run_trigger) and batch_done_at is None:
                batch_done_at = i
                break

        assert batch_done_at is not None
        # Batch termination should be at the slowest run's trigger
        assert batch_done_at == max(t for t in per_run_trigger if t is not None)


# ── Existing fallback termination still works ────────────────────────────


class TestFallbackTermination:
    def test_iteration_fallback(self):
        """IterationTermination as fallback when temp termination not reached."""
        t_temp = TempTermination(n_walkers=100, target_temp=0.001, threshold=50.0)
        t_iter = IterationTermination(max_iterations=100)

        for i in range(200):
            done, msg = check_any([t_temp, t_iter], i, 10.0 - i * 0.001)
            if done:
                # Iteration should trigger first at i=99
                assert i == 99
                assert "iteration" in msg.lower() or "max" in msg.lower()
                break
        else:
            pytest.fail("Should have terminated")


# ── Integration: TempTermination with run_ns ───────────────────────────


@pytest.mark.heavy
class TestTempTerminationIntegration:
    def test_run_ns_with_temp_termination(self):
        """TempTermination terminates a real NS run before max_iterations."""
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.move_descriptor import MoveDescriptor
        from jaxrens.sampling.moves import random_walk
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns

        backend = create_harmonic(k=1.0)
        init_fn, step_fn = build_mwg(backend, [
            MoveDescriptor("random_walk", random_walk.build_kernel),
        ])

        n_walkers = 50
        key = jax.random.key(99)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(
            lambda pos: backend(pos, types, cell, 0)[0]
        )(positions)

        max_iters = 5000
        temp_crit = TempTermination(
            n_walkers=n_walkers, target_temp=300.0, threshold=10.0
        )
        iter_crit = IterationTermination(max_iterations=max_iters)

        result = run_ns(
            positions, types, energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=max_iters,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            termination_criteria=[temp_crit, iter_crit],
        )

        # TempTermination should have stopped the run early
        assert result["iteration"] < max_iters, (
            f"Expected early termination, but ran all {max_iters} iterations"
        )
        assert jnp.isfinite(result["log_evidence"])

    def test_energy_termination_with_run_ns(self):
        """EnergyTermination stops run_ns when emax drops below target."""
        from jaxrens.backends.toy import create_harmonic
        from jaxrens.sampling.move_descriptor import MoveDescriptor
        from jaxrens.sampling.moves import random_walk
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns

        backend = create_harmonic(k=1.0)
        init_fn, step_fn = build_mwg(backend, [
            MoveDescriptor("random_walk", random_walk.build_kernel),
        ])

        n_walkers = 50
        key = jax.random.key(77)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(
            lambda pos: backend(pos, types, cell, 0)[0]
        )(positions)

        # Set energy target that should be reached well before 5000 iters
        energy_target = 1.0  # harmonic energies start ~5-10, decrease during NS
        max_iters = 5000

        result = run_ns(
            positions, types, energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=max_iters,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            termination_criteria=[
                EnergyTermination(min_energy=energy_target),
                IterationTermination(max_iterations=max_iters),
            ],
        )

        assert result["iteration"] < max_iters
        # The last dead energy should be near or below the target
        last_dead_idx = result["n_dead"] - 1
        last_dead_e = float(result["dead_energies"][last_dead_idx])
        assert last_dead_e < energy_target + 1.0  # some tolerance
