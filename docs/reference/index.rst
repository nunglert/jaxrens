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

.. note::

   ``import jaxrens`` exposes only ``__version__`` (and does not import
   JAX, see :doc:`../dev/contributing`).  Import public symbols from their
   subpackage — ``from jaxrens.sampling.nested_sampling import run_ns`` —
   and browse the :ref:`package tree <subpackages>` below to find them.

.. _subpackages:

Subpackages and modules
-----------------------

For the internal organization (move kernels, adaptation, init
helpers, constraints, I/O writers), browse the subpackages and
top-level modules directly:

.. autosummary::
   :toctree: api/
   :recursive:

   backends
   sampling
   state
   init
   constraints
   io
   postprocess
   cli
   utils
   prior_mass
   unvalidated
