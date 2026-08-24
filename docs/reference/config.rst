Configuration reference
=======================

Every ``jaxrens`` run is driven by a YAML configuration file that is parsed
and validated against a set of `Pydantic <https://docs.pydantic.dev/latest/>`_
models before any sampling begins.  This page documents **every allowed
parameter** — its type, default, and constraints — generated directly from
those models, so it can never drift from the code that enforces it.

The top-level document is :class:`~jaxrens.cli.schema.root.RootSpec`.  Its
``moves``, ``backend``, ``ensemble`` and ``termination`` fields are
*discriminated unions*: the ``type:`` key in the YAML selects which concrete
model applies, and each variant is documented in its own section below.


Root document
-------------

.. autopydantic_model:: jaxrens.cli.schema.root.RootSpec
   :model-show-json: true
   :field-show-constraints: true

.. note::

   ``interval_units`` rescales every iteration-counted field at once
   (``per_walker`` counts in walker-sweeps of ``run.n_live`` iterations
   instead of raw iterations).  See the "Interval-unit scaling" section
   of :doc:`../user/concepts/schema_resolve` for the motivation, a YAML
   example, and the full list of affected fields.


Run settings
------------

.. autopydantic_model:: jaxrens.cli.schema.run.RunSpec
   :field-show-constraints: true


Moves
-----

The ``moves:`` key takes a list of move specifications (a single mapping is
also accepted and wrapped in a list).  Each entry's ``type:`` selects one of
the variants below.

.. autopydantic_model:: jaxrens.cli.schema.moves.BaseMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.RandomWalkMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.GMCMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.HMCMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.SingleAtomMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.SingleAtomSweepMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.VolumeMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.ShearMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.StretchMoveSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.moves.AlchemicalMorphMoveSpec
   :field-show-constraints: true


Backends
--------

The ``backend:`` key selects the energy model.  Its ``type:`` chooses one of
the variants below.

.. autopydantic_model:: jaxrens.cli.schema.backend.BaseBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.HarmonicBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.DoubleWellBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.GaussianMixtureBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.LJBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.NeuralILBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.MACEBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.NequixBackendSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.backend.JaxMDBackendSpec
   :field-show-constraints: true


Ensemble
--------

The ``ensemble:`` key defaults to NVT; ``type: npt`` enables constant-pressure
sampling.

.. autopydantic_model:: jaxrens.cli.schema.ensemble.NVTEnsembleSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.ensemble.NPTEnsembleSpec
   :field-show-constraints: true


Termination
-----------

The ``termination:`` key takes a list of stopping criteria (a single mapping
is wrapped in a list); the run stops when any of them fires.

.. autopydantic_model:: jaxrens.cli.schema.termination.BaseTerminationSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.termination.IterationTerminationSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.termination.PriorMassTerminationSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.termination.TemperatureTerminationSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.termination.EnergyTerminationSpec
   :field-show-constraints: true


Adaptation
----------

.. autopydantic_model:: jaxrens.cli.schema.adaptation.AdaptationSpec
   :field-show-constraints: true


Initialization
--------------

The ``init:`` section selects the source of atoms (exactly one of
``start_species``, ``start_config_file``, ``start_walker_set`` or
``restart_file``).

.. autopydantic_model:: jaxrens.cli.schema.init.InitSpec
   :field-show-constraints: true

.. autopydantic_model:: jaxrens.cli.schema.init.InitialWalkSpec
   :field-show-constraints: true


Simulation cell
---------------

.. autopydantic_model:: jaxrens.cli.schema.cell.CellSpec
   :field-show-constraints: true


Output
------

.. autopydantic_model:: jaxrens.cli.schema.output.OutputSpec
   :field-show-constraints: true


Inter-replica exchange
----------------------

The ``inter_re:`` key is optional; omitting it disables replica-exchange swaps
entirely.

.. autopydantic_model:: jaxrens.cli.schema.inter_re.InterRESpec
   :field-show-constraints: true


Constraints
-----------

The ``constraints:`` key takes a list of hard configuration constraints (a
single mapping is wrapped in a list).  Each entry's ``type:`` selects one of
the variants below.  A constraint rejects any proposal that would move a
walker into a forbidden region, exactly like the likelihood threshold; it is
enforced only on the moves that can actually violate it (e.g. a
minimum-distance constraint gates atom-displacement and cell moves, and gates
species-changing moves only when its thresholds vary by element pair).

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

.. autopydantic_model:: jaxrens.cli.schema.constraints.MinDistanceConstraintSpec
   :field-show-constraints: true
