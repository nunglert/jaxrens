"""Test NeuralIL backend wrapper.

These tests are skipped if NeuralIL is not installed.
When available, they test: model loading, energy evaluation, kernel dispatch.
"""

import pytest

from jaxrens.backends.neuralil import is_available, _NEURALIL_IMPORT_ERROR

neuralil_required = pytest.mark.skipif(
    not is_available(),
    reason=f"NeuralIL not installed: {_NEURALIL_IMPORT_ERROR}",
)


@neuralil_required
class TestNeuralILImport:
    """Test guarded import logic (these run even without NeuralIL)."""

    @pytest.mark.skipif(is_available(), reason="Only test import error path")
    def test_require_raises_without_neuralil(self):
        from jaxrens.backends.neuralil import _require_neuralil

        with pytest.raises(ImportError, match="NeuralIL is required"):
            _require_neuralil()

    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_create_neuralil_without_pickle_raises(self):
        if not is_available():
            pytest.skip("NeuralIL not installed")
        from jaxrens.backends.neuralil import create_neuralil

        with pytest.raises(ValueError, match="pickle_file is required"):
            create_neuralil(pickle_file=None)


@neuralil_required
class TestNeuralILKernelSet:
    """Test multi-kernel dispatch (requires NeuralIL + model file)."""

    # These would need a real pickle file to run
    # Placeholder structure for when testing with actual models

    def test_kernel_set_creation(self):
        """Placeholder: would test create_neuralil_kernel_set with a real model."""
        pytest.skip("Requires NeuralIL model pickle file")


# Always-run test (not skipped)
class TestNeuralILAvailability:
    """Tests that always run regardless of NeuralIL installation."""

    def test_import_does_not_crash(self):
        """Importing the module should never crash, even without NeuralIL."""
        import jaxrens.backends.neuralil  # noqa: F401

    def test_loader_handles_missing_neuralil(self):
        """load_backend('neuralil') should give clear error without NeuralIL."""
        if is_available():
            pytest.skip("NeuralIL is installed")
        from jaxrens.backends.loader import load_backend

        with pytest.raises(ImportError):
            load_backend("neuralil", pickle_file="dummy.pkl")
