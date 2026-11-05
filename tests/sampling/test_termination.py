"""Tests for the termination criteria and check_any."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.prior_mass import log_alpha, log_dead_weights
from jaxrens.sampling.batch_descriptor import VmapRuns
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import run_ns
from jaxrens.sampling.termination import (
    EnergyTermination,
    IterationTermination,
    PriorMassTermination,
    TempTermination,
    check_any,
)


class TestIterationTermination:
    """Check that IterationTermination stops at the last iteration."""

    def test_triggers_at_max(self):
        t = IterationTermination(max_iterations=100)
        assert not t.check(98, 0.0)
        assert t.check(99, 0.0)

    def test_message(self):
        t = IterationTermination(max_iterations=10)
        assert "10" in t.message()


class TestPriorMassTermination:
    """Check PriorMassTermination with and without an evidence estimate."""

    def test_never_triggers_without_evidence(self):
        """Without update_evidence the log-evidence is -inf: never triggers."""
        t = PriorMassTermination(n_live=100)
        assert not t.check(0, 0.0)
        assert not t.check(50, 0.0)

    def test_triggers_when_mass_negligible(self):
        t = PriorMassTermination(n_live=50, threshold=0.1)
        t.update_evidence(-5.0)
        # At iteration 1000, log_remaining = -1000/50 = -20, below -5.1.
        assert t.check(1000, 0.0)


class TestNCullAwareTermination:
    """Check that the criteria use the n_cull ladder of jaxrens.prior_mass."""

    def test_prior_mass_uses_n_cull(self):
        n, c = 50, 5
        t = PriorMassTermination(n_live=n, threshold=1.0, n_cull=c)
        # log X after iteration i is (i + 1) * log_alpha.  With log_Z = 0 and
        # emax = 0 it triggers once that is below log(threshold) = 0, right
        # after the guard that every walker is replaced once.
        t.update_evidence(0.0)
        assert not t.check(n // c, 0.0)  # guard: i * c <= n
        assert t.check(n // c + 1, 0.0)
        assert t._log_alpha == pytest.approx(log_alpha(n, c))

    def test_prior_mass_n_cull_contracts_faster(self):
        slow = PriorMassTermination(n_live=100, threshold=1e-3, n_cull=1)
        fast = PriorMassTermination(n_live=100, threshold=1e-3, n_cull=10)
        first = {}
        for name, t in (("slow", slow), ("fast", fast)):
            t.update_evidence(-1.0)
            first[name] = next(i for i in range(10_000) if t.check(i, 0.0))
        assert first["fast"] * 8 < first["slow"]

    def test_temp_termination_weight_is_contour_dead_point(self):
        n, c, i = 40, 4, 17
        t = TempTermination(n_walkers=n, target_temp=1000.0, n_cull=c)
        t.check(i, 0.0)  # emax = 0: log_Z_term is the bare weight
        expected = log_dead_weights((i + 1) * c, n, c)[-1]
        assert t.log_Z_term_last == pytest.approx(expected)


class TestEnergyTermination:
    """Check that EnergyTermination stops when emax drops below the target."""

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


class TestTempTermination:
    """Check TempTermination on synthetic sequences of dead-point energies.

    The tests cover convergence, the threshold, n_cull, the log-term
    properties and the message.
    """

    def test_basic_convergence(self):
        """Exponentially decaying energies trigger the criterion."""
        n_walkers = 100
        target_temp = 300.0
        t = TempTermination(
            n_walkers=n_walkers, target_temp=target_temp, threshold=10.0
        )

        n_iters = 5000
        energies = [
            10.0 * math.exp(-i / (n_walkers * 0.5)) for i in range(n_iters)
        ]

        triggered_at = None
        for i, e in enumerate(energies):
            if t.check(i, e):
                triggered_at = i
                break

        assert triggered_at is not None
        assert triggered_at > 0

    def test_triggers_with_flat_energies(self):
        """A constant energy still triggers once the prior-mass weight decays.

        With a constant energy, log_Z_term only falls through the weight term
        i * log_alpha, so it drops below the maximum after enough iterations.
        """
        t = TempTermination(n_walkers=50, target_temp=1000.0, threshold=10.0)
        triggered = False
        for i in range(10000):
            if t.check(i, 5.0):
                triggered = True
                break
        assert triggered

    def test_n_cull_greater_than_one(self):
        """The criterion also triggers with n_cull = 3."""
        t = TempTermination(
            n_walkers=100, target_temp=500.0, n_cull=3, threshold=10.0
        )

        triggered_at = None
        for i in range(5000):
            e = 10.0 * math.exp(-i / 50.0)
            if t.check(i, e):
                triggered_at = i
                break

        assert triggered_at is not None

    def test_threshold_sensitivity(self):
        """A smaller threshold triggers earlier."""
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
        assert trigger_5 < trigger_15

    def test_log_Z_term_properties(self):
        """log_Z_term_max and log_Z_term_last start at -inf and then update."""
        t = TempTermination(n_walkers=100, target_temp=300.0)
        assert t.log_Z_term_max == -math.inf
        assert t.log_Z_term_last == -math.inf

        t.check(0, 5.0)
        assert t.log_Z_term_max > -math.inf
        assert t.log_Z_term_last > -math.inf

    def test_message_format(self):
        """The message includes the temperature and the word converged."""
        t = TempTermination(n_walkers=100, target_temp=300.0)
        for i in range(3000):
            if t.check(i, 10.0 * math.exp(-i / 50.0)):
                break
        msg = t.message()
        assert "300.0" in msg
        assert "converged" in msg


class TestCheckAny:
    """Check that check_any returns the first criterion that is satisfied."""

    def test_returns_first_satisfied(self):
        c1 = IterationTermination(max_iterations=1000)
        c2 = EnergyTermination(min_energy=5.0)
        done, msg = check_any([c1, c2], iteration=10, emax=3.0)
        assert done
        assert msg == c2.message()

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
        assert msg == c1.message()


class CustomCriterion:
    """Custom criterion that stops when the energy oscillates."""

    def __init__(self, max_oscillations: int = 3):
        self.max_oscillations = max_oscillations
        self._prev_energy = None
        self._prev_direction = None
        self._oscillations = 0

    def check(self, iteration: int, emax: float) -> bool:
        if self._prev_energy is not None:
            direction = emax > self._prev_energy
            if (
                self._prev_direction is not None
                and direction != self._prev_direction
            ):
                self._oscillations += 1
            self._prev_direction = direction
        self._prev_energy = emax
        return self._oscillations >= self.max_oscillations

    def message(self) -> str:
        return f"Energy oscillated {self._oscillations} times"


class TestPluggableProtocol:
    """Check that a custom criterion works with check_any."""

    def test_custom_criterion_with_check_any(self):
        custom = CustomCriterion(max_oscillations=2)
        criteria = [IterationTermination(max_iterations=1000), custom]

        # Oscillations occur at indices 2, 3 and 4.
        energies = [5.0, 3.0, 6.0, 2.0, 7.0, 1.0, 8.0]
        for i, e in enumerate(energies):
            done, msg = check_any(criteria, i, e)
            if done:
                assert "oscillat" in msg.lower()
                break
        else:
            pytest.fail("Custom criterion should have triggered")


class TestBatchTermination:
    """Check TempTermination on parallel runs and the batch reduction."""

    def test_all_runs_must_converge(self):
        """Four runs with different schedules converge at different times."""
        n_runs = 4
        criteria = [
            TempTermination(n_walkers=100, target_temp=300.0, threshold=10.0)
            for _ in range(n_runs)
        ]

        schedules = [
            [10.0 * math.exp(-i / 50.0) for i in range(5000)],
            [8.0 * math.exp(-i / 60.0) for i in range(5000)],
            [12.0 * math.exp(-i / 40.0) for i in range(5000)],
            [6.0 * math.exp(-i / 80.0) for i in range(5000)],
        ]

        all_triggered = [None] * n_runs
        for i in range(5000):
            for r in range(n_runs):
                if all_triggered[r] is None and criteria[r].check(
                    i, schedules[r][i]
                ):
                    all_triggered[r] = i

            if all(t is not None for t in all_triggered):
                break

        assert all(t is not None for t in all_triggered)
        assert len(set(all_triggered)) > 1

    def test_slowest_run_determines_batch_termination(self):
        """The batch criterion fires when the slowest run converges.

        VmapRuns.reduce_for_termination takes the worst case across runs, the
        maximum energy.  The slow run has the highest energy at every
        iteration, so the batch criterion sees the slow run's energies and
        fires at the same iteration as the slow run's own criterion.
        """
        n_runs = 3
        n_iters = 5000
        batcher = VmapRuns(n_runs=n_runs)
        run_criteria = [
            TempTermination(n_walkers=100, target_temp=300.0, threshold=10.0)
            for _ in range(n_runs)
        ]
        batch_criterion = TempTermination(
            n_walkers=100, target_temp=300.0, threshold=10.0
        )
        decay_lengths = [40.0, 40.0, 100.0]  # two fast runs and one slow run

        run_trigger = [None] * n_runs
        batch_trigger = None
        for i in range(n_iters):
            energies = [10.0 * math.exp(-i / d) for d in decay_lengths]
            for r in range(n_runs):
                if run_trigger[r] is None and run_criteria[r].check(
                    i, energies[r]
                ):
                    run_trigger[r] = i
            _, hmax = batcher.reduce_for_termination(
                jnp.zeros(n_runs), jnp.array(energies)
            )
            if batch_criterion.check(i, hmax):
                batch_trigger = i
                break

        assert batch_trigger is not None
        assert batch_trigger == run_trigger[2]
        assert batch_trigger > run_trigger[0]


class TestFallbackTermination:
    """Check that IterationTermination stops a run that never converges."""

    def test_iteration_fallback(self):
        """IterationTermination stops the run when TempTermination does not."""
        t_temp = TempTermination(
            n_walkers=100, target_temp=0.001, threshold=50.0
        )
        t_iter = IterationTermination(max_iterations=100)

        for i in range(200):
            done, msg = check_any([t_temp, t_iter], i, 10.0 - i * 0.001)
            if done:
                assert i == 99
                assert "iteration" in msg.lower() or "max" in msg.lower()
                break
        else:
            pytest.fail("Should have terminated")


@pytest.mark.heavy
class TestTempTerminationIntegration:
    """Check that run_ns stops early with a temperature or energy criterion."""

    def test_run_ns_with_temp_termination(self):
        """TempTermination terminates a real NS run before max_iterations."""
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("random_walk", random_walk.build_kernel),
            ],
        )

        n_walkers = 50
        key = jax.random.key(99)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(lambda pos: backend(pos, types, cell, 0)[0])(
            positions
        )

        max_iters = 5000
        temp_crit = TempTermination(
            n_walkers=n_walkers, target_temp=300.0, threshold=10.0
        )
        iter_crit = IterationTermination(max_iterations=max_iters)

        result = run_ns(
            positions,
            types,
            energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=max_iters,
            n_mcmc_steps=5,
            initial_step_size=0.3,
            termination_criteria=[temp_crit, iter_crit],
        )

        assert result["iteration"] < max_iters
        assert jnp.isfinite(result["log_evidence"])

    def test_energy_termination_with_run_ns(self):
        """EnergyTermination stops run_ns when emax drops below target."""
        backend = create_harmonic(k=1.0)
        init_fn, step_fn, _ = build_mwg(
            backend,
            [
                MoveKernel("random_walk", random_walk.build_kernel),
            ],
        )

        n_walkers = 50
        key = jax.random.key(77)
        key, init_key = jax.random.split(key)

        positions = jax.random.uniform(
            init_key, (n_walkers, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        cell = jnp.zeros((3, 3))
        energies = jax.vmap(lambda pos: backend(pos, types, cell, 0)[0])(
            positions
        )

        # Harmonic energies start around 5-10 and decrease during NS.
        energy_target = 1.0
        max_iters = 5000

        result = run_ns(
            positions,
            types,
            energies,
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
        # EnergyTermination fires when the worst live energy is below the
        # target, so every surviving walker sits below it too.
        assert float(jnp.max(result["energies"])) < energy_target


class TestTerminationEndToEndJit:
    """Check that a termination spec stops run_ns, which jits ns_step."""

    def test_iteration_termination_stops_early_under_jit(self):
        """IterationTermination(5) makes run_ns stop at iteration 5."""
        d = {
            "run": {
                "n_live": 10,
                "max_iterations": 1000,
                "n_mcmc_steps": 3,
                "seed": 0,
            },
            "moves": [{"move_type": "random_walk", "step_size": 0.3}],
            "backend": {"backend_type": "harmonic"},
            "output": {
                "format": "none",
                "working_dir": ".",
                "info_interval": 999,
            },
            "termination": [{"type": "iteration", "max_iterations": 5}],
        }
        root = RootSpec.model_validate(d)
        resolved = resolve(root)

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(
            backend, list(resolved.move_descriptors)
        )

        key = jax.random.key(123)
        key, key_pos = jax.random.split(key)
        n_live = 10
        positions = jax.random.uniform(
            key_pos, (n_live, 1, 3), minval=-2.0, maxval=2.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        result = run_ns(
            positions=positions,
            types=types,
            energies=energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_key=key,
            max_iterations=1000,
            n_mcmc_steps=3,
            termination_criteria=list(resolved.termination),
        )

        assert result["iteration"] <= 6
        assert result["n_dead"] <= 6
