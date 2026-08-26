API reference
=============

The public surface of ``jaxrens``, organised by subpackage.  Each symbol
is linked to its own page with the full docstring; import symbols from
their subpackage, e.g. ``from jaxrens.sampling.nested_sampling import
run_ns``.  The CLI subcommands are documented separately at :doc:`cli`.
Every allowed YAML configuration parameter is documented at :doc:`config`.

An interactive overview of the package layout — subpackages and
modules sized by lines of code — sets the stage:

.. only:: html

   .. raw:: html

      <iframe
          src="../_static/figures/pkg_treemap.html"
          width="100%"
          height="620"
          style="border: 1px solid #e1e1e1; border-radius: 4px; background: white;"
          loading="lazy"
          title="jaxrens package treemap (interactive)"
      ></iframe>

.. only:: latex

   .. image:: /_static/figures/pkg_treemap.svg
      :alt: treemap of the jaxrens package — subpackages and modules sized by LoC
      :align: center
      :width: 100%

.. currentmodule:: jaxrens

.. important::

   There is no curated top-level API right now: ``import jaxrens`` exposes
   only ``__version__``.  Import public symbols from their subpackage —
   ``from jaxrens.sampling.nested_sampling import run_ns`` — and browse the
   :ref:`Subpackages <subpackages>` tree below to find them.

.. dropdown:: Maintainer note — reinstating the top-level API
   :class-title: sd-fs-6

   The curated top-level API (``from jaxrens import run_ns``,
   ``jaxrens.free_energy``, …) is disabled; see the commented lazy-export
   block in ``src/jaxrens/__init__.py``.  Keeping it off is deliberate:
   ``import jaxrens`` must not import JAX (see :doc:`../dev/contributing`),
   and a lazy re-export surface is easy to break that rule with.

   When it was enabled this page listed these symbols grouped by theme:

   * *High-level entry points:* ``run_ns``, ``run_from_config``,
     ``init_ns``, ``ns_step``, ``build_mwg``, ``load_backend``
   * *State and configuration:* ``MoveKernel``, ``MCState``,
     ``make_mc_state_class``, ``NSConfig``, ``MoveConfig``,
     ``BackendConfig``, ``OutputConfig``
   * *Backend interfaces:* ``EnergyBackend``, ``EnsembleBackend``,
     ``make_ensemble_params``
   * *Post-processing:* ``calc_log_weights``, ``calc_log_weights_live``,
     ``log_evidence``, ``partition_function``, ``heat_capacity``,
     ``expectation``, ``free_energy``

   To reinstate, uncomment the lazy-export block in
   ``src/jaxrens/__init__.py`` and restore the four ``autosummary``
   sections here (recoverable from git history).

.. _subpackages:

Subpackages
-----------

For the internal organization (move kernels, adaptation, init
helpers, I/O writers), browse the subpackages directly:

.. autosummary::
   :toctree: api/
   :recursive:

   backends
   sampling
   state
   init
   io
   postprocess
   cli
