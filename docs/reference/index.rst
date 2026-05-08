API reference
=============

The public surface of ``jaxrens`` — everything importable from the
top-level package.  Each symbol is linked to its own page with the
full docstring; the CLI subcommands are documented separately at
:doc:`cli`.

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

High-level entry points
-----------------------

.. autosummary::
   :toctree: api/
   :nosignatures:

   run_ns
   run_from_config
   init_ns
   ns_step
   build_mwg
   load_backend

State and configuration
-----------------------

.. autosummary::
   :toctree: api/
   :nosignatures:

   MoveKernel
   MCState
   make_mc_state_class
   NSConfig
   MoveConfig
   BackendConfig
   OutputConfig

Backend interfaces
------------------

.. autosummary::
   :toctree: api/
   :nosignatures:

   EnergyBackend
   EnsembleBackend
   make_ensemble_params

Post-processing
---------------

.. autosummary::
   :toctree: api/
   :nosignatures:

   calc_log_weights
   calc_log_weights_live
   log_evidence
   partition_function
   heat_capacity
   expectation
   free_energy

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
