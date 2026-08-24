"""Tests for the unvalidated-feature marker (``jaxrens.unvalidated``).

The marker is about *production validation*, not test coverage, so these
tests check the mechanism (registry, one-shot warning, env policy, docstring
admonition, output stamping) rather than any physics.
"""

from __future__ import annotations

import warnings

import pytest

from jaxrens.unvalidated import (
    REGISTRY,
    Unvalidated,
    UnvalidatedFeatureWarning,
    format_summary,
    reset,
    triggered,
    unvalidated,
    warn_unvalidated,
)


@pytest.fixture(autouse=True)
def _fresh_markers(monkeypatch):
    """One-shot suppression is process-global; clear it around every test."""
    monkeypatch.delenv("JAXRENS_UNVALIDATED", raising=False)
    reset()
    yield
    reset()


def _make(**kw):
    """A freshly-decorated function; its registry key is on the wrapper."""
    kw.setdefault("concern", "never run on a real system")
    kw.setdefault("since", "0.2.2")

    @unvalidated(**kw)
    def fn(a, b=2):
        """Summary line.

        Args:
            a: first.
        """
        return a + b

    return fn


class TestDecorator:
    def test_registers_at_import_without_calling(self):
        fn = _make()
        key = fn.__jaxrens_unvalidated__.feature
        assert key in REGISTRY
        assert key.endswith(".fn")
        assert REGISTRY[key].concern == "never run on a real system"
        # Declared, but not yet triggered -- nobody called it.
        assert triggered() == ()

    def test_warns_on_first_call_only(self):
        fn = _make()
        with pytest.warns(
            UnvalidatedFeatureWarning, match="never run on a real"
        ):
            assert fn(1) == 3
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fn(1) == 3  # second call is silent

    def test_preserves_signature_and_return(self):
        fn = _make()
        import inspect

        assert list(inspect.signature(fn).parameters) == ["a", "b"]
        with pytest.warns(UnvalidatedFeatureWarning):
            assert fn(1, b=10) == 11

    def test_record_exposed_on_wrapper(self):
        fn = _make()
        rec = fn.__jaxrens_unvalidated__
        assert isinstance(rec, Unvalidated)
        assert rec.since == "0.2.2"

    def test_docstring_gets_admonition(self):
        fn = _make(clears_when="a 512-atom run")
        assert "Summary line." in fn.__doc__
        assert ".. warning::" in fn.__doc__
        assert "**Unvalidated** (since 0.2.2)" in fn.__doc__
        assert "*Cleared by:* a 512-atom run" in fn.__doc__
        # Body of the original docstring survives intact.
        assert "a: first." in fn.__doc__

    def test_admonition_on_undocumented_function(self):
        @unvalidated(concern="c", since="0.2.2")
        def bare():
            pass

        assert bare.__doc__.startswith(".. warning::")


class TestPolicy:
    def test_ignore_is_silent_but_still_records(self, monkeypatch):
        monkeypatch.setenv("JAXRENS_UNVALIDATED", "ignore")
        fn = _make()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fn(1)
        # Silent, but the run is still stamped -- that is the point.
        assert any(r.feature.endswith(".fn") for r in triggered())

    def test_error_raises_every_time(self, monkeypatch):
        monkeypatch.setenv("JAXRENS_UNVALIDATED", "error")
        fn = _make()
        for _ in range(2):
            with pytest.raises(UnvalidatedFeatureWarning, match="never run"):
                fn(1)

    def test_policy_read_per_call_not_cached(self, monkeypatch):
        fn = _make()
        with pytest.warns(UnvalidatedFeatureWarning):
            fn(1)
        reset()
        monkeypatch.setenv("JAXRENS_UNVALIDATED", "error")
        with pytest.raises(UnvalidatedFeatureWarning):
            fn(1)

    def test_invalid_policy_falls_back_to_warn(self, monkeypatch):
        monkeypatch.setenv("JAXRENS_UNVALIDATED", "nonsense")
        import jaxrens.unvalidated as mod

        monkeypatch.setattr(mod, "_WARNED_BAD_POLICY", False)
        fn = _make()
        with pytest.warns((UnvalidatedFeatureWarning, RuntimeWarning)):
            fn(1)


class TestCallForm:
    def test_warn_unvalidated_registers_and_warns(self):
        with pytest.warns(UnvalidatedFeatureWarning, match=">2 species"):
            warn_unvalidated(
                "swap: >2 species", concern="only binaries run", since="0.2.2"
            )
        assert "swap: >2 species" in REGISTRY
        assert triggered()[-1].feature == "swap: >2 species"

    def test_format_summary(self):
        with pytest.warns(UnvalidatedFeatureWarning):
            warn_unvalidated("feat.x", concern="unchecked", since="0.2.2")
        assert "feat.x (since 0.2.2): unchecked" in format_summary()

    def test_message_names_the_escape_hatches(self):
        with pytest.warns(UnvalidatedFeatureWarning) as rec:
            warn_unvalidated("feat.y", concern="unchecked", since="0.2.2")
        msg = str(rec[0].message)
        assert "JAXRENS_UNVALIDATED=ignore" in msg
        assert "=error" in msg


class TestHmcMarker:
    def test_hmc_build_kernel_is_marked(self):
        from jaxrens.sampling.moves import hmc

        rec = hmc.build_kernel.__jaxrens_unvalidated__
        assert "delta_H" in rec.concern
        assert rec.since == "0.2.2"

    def test_hmc_build_kernel_warns_when_built(self):
        from jaxrens.sampling.moves import hmc

        with pytest.warns(UnvalidatedFeatureWarning, match="hmc.build_kernel"):
            hmc.build_kernel(backend=object(), n_leapfrog=3)


class TestTrajectoryStamping:
    def test_h5_writer_stamps_triggered_markers(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from jaxrens.io.trajectory import H5TrajectoryWriter

        with pytest.warns(UnvalidatedFeatureWarning):
            warn_unvalidated("feat.z", concern="unchecked", since="0.2.2")

        path = tmp_path / "traj.h5"
        writer = H5TrajectoryWriter(path, symbol_map={0: "Si"})
        writer.close()

        with h5py.File(path, "r") as f:
            stamped = [str(x) for x in f.attrs["unvalidated_features"]]
        assert any("feat.z" in s for s in stamped)

    def test_no_attr_when_nothing_triggered(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from jaxrens.io.trajectory import H5TrajectoryWriter

        path = tmp_path / "clean.h5"
        writer = H5TrajectoryWriter(path, symbol_map={0: "Si"})
        writer.close()

        with h5py.File(path, "r") as f:
            assert "unvalidated_features" not in f.attrs


def test_module_does_not_import_jax():
    """The stamping helper is reachable from JAX-free writer paths."""
    import subprocess
    import sys

    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jaxrens.unvalidated, sys; "
            "assert 'jax' not in sys.modules, "
            "sorted(m for m in sys.modules if 'jax' in m)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
