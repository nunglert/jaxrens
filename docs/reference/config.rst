Configuration
=============

Every ``jaxrens`` run is driven by a single YAML file, validated against a set
of `Pydantic <https://docs.pydantic.dev/latest/>`_ models before any sampling
begins.  This page is the working reference for that file: every key you can
write, grouped by the section it lives in, with its type, default, and what it
does.  The tables are generated from the same models that validate your
config, so they cannot drift from the code that enforces them.

Validate a file without running anything:

.. code-block:: bash

   jaxrens validate -c config.yaml                # full check, imports JAX
   jaxrens validate --parse-only -c config.yaml   # schema only, fast
   jaxrens dump-schema                            # the raw JSON Schema

.. seealso::

   :doc:`../user/concepts/schema_resolve` explains how the parsed config
   becomes a running job.  The pydantic models themselves live in
   :mod:`jaxrens.cli.schema`; ``jaxrens dump-schema`` emits the equivalent
   JSON Schema, descriptions included.


A complete config
-----------------

Everything below is one file.  ``run``, ``moves``, ``backend`` and ``output``
are required; the rest have defaults.

.. code-block:: yaml

   interval_units: absolute

   run:                          # sampler sizing
     n_live: 128
     n_mcmc_steps: 20
     seed: 42

   backend:                      # the energy model
     type: lj
     epsilon: 1.0
     sigma: 1.0
     cutoff: 2.5
     periodic: true

   ensemble:                     # optional; NVT if omitted
     type: npt
     pressure: 0.1
     pressure_units: eva3

   moves:                        # MWG kernels; weights set dispatch odds
     - {type: gmc, n_reflect: 10, step_size: 0.1, weight: 4.0}
     - {type: volume, step_size: 0.3, weight: 1.0}
     - {type: shear,  step_size: 0.1, weight: 1.0}
     - {type: stretch, step_size: 0.1, weight: 1.0}

   termination:                  # run stops when any criterion fires
     - type: prior_mass
       threshold: 1.e-5

   adaptation:                   # off unless adjust_interval is set
     adjust_interval: 10
     full_auto: true
     defaults: {min_rate: 0.3, max_rate: 0.5, step_size_max: 0.2}
     per_move:
       gmc: {step_size_max: 0.5}

   init:                         # exactly one source of atoms
     start_species: "18 64"      # 64 argon atoms
     pos_randomization_mode: grid
     grid_distance: 1.0
     initial_walk: {n_walks: 5, walklength: 50}

   cell:                         # bounds for the volume/shear/stretch moves
     max_volume_per_atom: 20.0
     min_volume_per_atom: 0.5
     min_aspect_ratio: 0.6

   constraints:                  # optional hard geometry constraints
     - type: minimum_distance
       d_min: 0.8

   inter_re:                     # optional; omit to disable replica exchange
     flavor: pressure
     re_interval: 1

   output:
     working_dir: ./output
     out_file_prefix: lj64_npt
     info_interval: 100
     checkpoint_interval: 500


Top-level keys
--------------

.. yaml-section:: jaxrens.cli.schema.root.RootSpec
   :no-docstring:

.. note::

   ``interval_units`` rescales every iteration-counted field at once
   (``per_walker`` counts in walker-sweeps of ``run.n_live`` iterations
   instead of raw iterations).  See the "Interval-unit scaling" section of
   :doc:`../user/concepts/schema_resolve` for the motivation, a YAML example,
   and the full list of affected fields.


``run:`` — sampler sizing
-------------------------

.. yaml-section:: jaxrens.cli.schema.run.RunSpec
   :prefix: run
   :no-docstring:


``moves:`` — MCMC kernels
-------------------------

A list of move specifications; a single mapping is accepted and wrapped in a
list.  Each entry's ``type:`` selects the kernel.  The scheduler composes them
into one Metropolis-within-Gibbs step, dispatching by ``weight``.

Every move accepts these shared keys:

.. yaml-section:: jaxrens.cli.schema.moves.BaseMoveSpec
   :prefix: moves[]
   :no-docstring:

And each ``type:`` adds its own:

.. yaml-variants:: jaxrens.cli.schema.moves.MoveSpec
   :prefix: moves[]
   :own-fields:

.. tip::

   Cell moves (``volume``, ``shear``, ``stretch``) take their bounds from the
   ``cell:`` section rather than from the move entry, and ``n_atoms`` comes
   from the resolved initial structure — that is why those tabs carry no
   geometry keys of their own.


``backend:`` — the energy model
-------------------------------

Shared keys, accepted by every backend:

.. yaml-section:: jaxrens.cli.schema.backend.BaseBackendSpec
   :prefix: backend
   :no-docstring:

Per-backend keys:

.. yaml-variants:: jaxrens.cli.schema.backend.BackendSpec
   :prefix: backend
   :own-fields:

``backend.softcore_repulsion:`` takes a mapping of:

.. yaml-section:: jaxrens.cli.schema.backend.SoftCoreSpec
   :prefix: backend.softcore_repulsion
   :no-docstring:


``ensemble:`` — thermodynamic ensemble
--------------------------------------

Defaults to NVT.  A **list-valued** driving parameter here (``pressure``, or
``chemical_potentials``) is what fans a run out across replicas — see
:doc:`../user/concepts/replicas`.

.. yaml-variants:: jaxrens.cli.schema.ensemble.EnsembleSpec
   :prefix: ensemble


``termination:`` — stopping criteria
------------------------------------

A list; the run stops when **any** entry fires.  A single mapping is wrapped
in a list.

.. yaml-variants:: jaxrens.cli.schema.termination.TerminationSpec
   :prefix: termination[]


``adaptation:`` — step-size adaptation
--------------------------------------

.. important::

   Adaptation is **on** by default: ``full_auto`` bisection every
   ``adjust_interval`` iterations.  Set ``adjust_interval: 0`` to switch it
   off and hold every move at its configured ``step_size``.

.. yaml-section:: jaxrens.cli.schema.adaptation.AdaptationSpec
   :prefix: adaptation
   :no-docstring:

Both ``defaults:`` and each entry of ``per_move:`` take the same policy
mapping.  Fields left unset fall through: ``per_move`` → ``defaults`` →
library fallback.

.. yaml-section:: jaxrens.cli.schema.adaptation.AdaptationPolicy
   :prefix: adaptation.defaults
   :no-docstring:


``init:`` — where the atoms come from
-------------------------------------

Exactly one of ``start_species``, ``start_config_file``, ``start_walker_set``
or ``restart_file`` must be set; setting zero or more than one is an error.

.. yaml-section:: jaxrens.cli.schema.init.InitSpec
   :prefix: init
   :no-docstring:
   :exclude: initial_walk

``init.initial_walk:`` configures the optional fixed-``E_max`` burn-in:

.. yaml-section:: jaxrens.cli.schema.init.InitialWalkSpec
   :prefix: init.initial_walk
   :no-docstring:


``cell:`` — cell-geometry bounds
--------------------------------

The single source of truth for the volume, shear, and stretch kernels; the
resolver threads these values into their ``kernel_kwargs``.

.. yaml-section:: jaxrens.cli.schema.cell.CellSpec
   :prefix: cell
   :no-docstring:


``output:`` — files, cadence, diagnostics
-----------------------------------------

.. yaml-section:: jaxrens.cli.schema.output.OutputSpec
   :prefix: output
   :no-docstring:


``inter_re:`` — inter-replica exchange
--------------------------------------

Optional.  Omitting the key disables replica-exchange swaps entirely.

.. yaml-section:: jaxrens.cli.schema.inter_re.InterRESpec
   :prefix: inter_re
   :no-docstring:


``constraints:`` — hard configuration constraints
-------------------------------------------------

A list; a single mapping is wrapped in a list.  A constraint rejects any
proposal that would move a walker into a forbidden region, exactly like the
likelihood threshold, and is enforced only on the moves that can violate it —
a minimum-distance constraint gates atom-displacement and cell moves, and
gates species-changing moves only when its thresholds vary by element pair.

Omitting the key (the default) means no constraints and zero overhead.

.. code-block:: yaml

   constraints:
     - type: minimum_distance
       d_min: 0.8                 # uniform floor (Angstrom)
     - type: minimum_distance
       d_min:                     # per-species-pair floors
         default: 1.0
         Si-Si: 2.0
         Si-O: 1.6

.. yaml-variants:: jaxrens.cli.schema.constraints.ConstraintSpec
   :prefix: constraints[]
