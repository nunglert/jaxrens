Output file reference
=====================

Every file a run writes, what is inside it, and how to read it back.

All of them share the ``output.out_file_prefix`` and land in
``output.working_dir`` — with one exception, the ``.log``, which is written
*beside* that directory rather than in it.  Nothing here is written
unconditionally: trajectories follow ``output.format``, the diagnostic logs
are opt-in, and the per-replica variants only appear for a multi-replica run.

.. tip::

   Four of these formats have a plot built in — ``jaxrens plot <file>``
   dispatches on the suffix and needs no Python.  See :doc:`cli`.


At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 14 22 34

   * - File
     - Format
     - Written when
     - Holds
   * - ``<prefix>.energies``
     - text
     - always
     - the dead-point ladder — the input to every estimator
   * - ``<prefix>.traj.extxyz`` / ``.traj.h5``
     - extxyz / HDF5
     - ``output.format``
     - the culled walker at each iteration
   * - ``<prefix>.traj.snap.<iter>.extxyz``
     - extxyz
     - ``snapshot_interval``
     - the whole live population, for crash inspection
   * - ``<prefix>.checkpoint.h5``
     - HDF5
     - ``checkpoint_interval``
     - full restartable state
   * - ``<prefix>.initial.checkpoint.h5``
     - HDF5
     - always
     - state at iteration 0, before any sampling
   * - ``<prefix>.final.checkpoint.h5``
     - HDF5
     - on clean exit
     - state at termination
   * - ``<prefix>.config.snapshot.yaml``
     - YAML
     - always
     - the fully-defaulted config this run used
   * - ``<prefix>.log``
     - text
     - always
     - progress and diagnostics (**beside** ``working_dir``)
   * - ``<prefix>.adaptation.h5``
     - HDF5
     - adaptation active
     - per-move step sizes and acceptance
   * - ``<prefix>.acc_rates.h5``
     - HDF5
     - ``save_acc_rates``
     - per-move accept/propose counts
   * - ``<prefix>.max_neighbors.h5``
     - HDF5
     - ``save_max_neighbors``
     - observed neighbour counts vs bucket size
   * - ``<prefix>.re_stats.h5``
     - HDF5
     - ``save_re_stats``
     - per-pair swap statistics

For a multi-replica run the per-replica files gain a ``.runNN.`` infix —
``<prefix>.run00.energies``, ``<prefix>.run01.traj.extxyz`` — numbered in
replica order.  The aggregate logs (``adaptation``, ``acc_rates``,
``max_neighbors``, ``re_stats``) do not: they carry an ``n_runs`` axis
instead.


``.energies`` — the dead-point ladder
-------------------------------------

Plain text, one row per culled walker, written as the run proceeds.  This is
the file thermodynamic post-processing consumes; everything else is
diagnostic.

The first line is a header::

    <n_walkers> <n_cull> <n_dof> 0.0 <n_atoms>

and every following line is one dead point::

    <iteration> <energy> <volume>

``volume`` is zero for runs without a cell.  Read it with
:class:`~jaxrens.io.energy_log.EnergyLogger`:

.. code-block:: python

   from jaxrens.io.energy_log import EnergyLogger

   log = EnergyLogger.read("output/ns.energies")
   log.iterations, log.energies, log.volumes    # numpy arrays
   log.n_walkers, log.n_cull, log.n_atoms       # from the header

Because it is append-only text, a truncated file from a killed job is still
readable up to the last complete line.


Trajectories
------------

``output.format`` selects the writer: ``extxyz`` (ASE-readable text),
``h5`` (compact binary, better for long runs), or ``none`` to write no
trajectory at all.

``<prefix>.traj.extxyz`` holds the **culled** walker at each iteration — the
same configurations the ``.energies`` ladder scores, so row *i* of one
corresponds to frame *i* of the other.  Written every
``output.traj_interval``; leave that at ``1`` if you intend to post-process,
since the estimators assume every dead point is present.

``<prefix>.traj.snap.<iter>.extxyz`` is different: a snapshot of the **entire
live population** at one iteration, for looking at a run that is misbehaving.
With ``output.snapshot_clean`` (the default) only the most recent survives.

``output.wrap_atoms`` controls whether positions are wrapped into each frame's
cell before writing.  It defaults to true because move kernels work in
unwrapped Cartesians and atoms otherwise drift arbitrarily far from the cell.

.. code-block:: python

   from ase.io import read

   frames = read("output/ns.traj.extxyz", index=":")   # every dead point
   last = read("output/ns.traj.extxyz", index=-1)


Checkpoints
-----------

HDF5, and the only files that can restart a run.  Three are written:
``.initial.checkpoint.h5`` before sampling begins, ``.checkpoint.h5``
refreshed every ``output.checkpoint_interval``, and ``.final.checkpoint.h5``
on clean termination.

Each holds the full live population (``positions``, ``types``, ``energies``,
and ``cells`` when present), per-move ``step_sizes`` so adaptation resumes
where it left off, and run-level state as attributes (``iteration``,
``n_walkers``, ``symbol_map``, log-evidence and counters).

Restart from one with ``init.restart_file``, or let ``jaxrens run --resume``
find the newest automatically.  A compatibility validator rejects a
checkpoint whose config disagrees with the one you are resuming under, rather
than continuing from an inconsistent state — see
:doc:`../user/concepts/restart`.


``.config.snapshot.yaml``
-------------------------

The validated config with **every default filled in**, dumped at startup.
Worth more than it looks: it records the values the run actually used,
including the ones you never wrote down, which is what makes a result
reproducible after a default changes upstream.  Diff it against your input to
see what the schema supplied.


Diagnostic logs
---------------

All HDF5, all with an ``iterations`` dataset of shape ``(N,)`` giving the NS
iteration each entry was recorded at, and an ``n_runs`` axis so one file
covers every replica.

``<prefix>.adaptation.h5``
    Written whenever step-size adaptation is active.  ``step_sizes`` and
    ``acceptance_rates``, both ``(N, n_runs, n_moves)`` float32, plus
    ``n_evaluations`` / ``n_grad_evaluations`` int64 and
    ``reject_reason_counts`` of shape ``(N, n_runs, n_moves, 4)`` — the four
    buckets being accepted, likelihood, cell-geometry and prior rejection.
    Read with :class:`~jaxrens.io.adaptation_log.AdaptationLogger`.

``<prefix>.acc_rates.h5``
    Opt-in via ``output.save_acc_rates``.  Raw ``n_accepted`` and
    ``n_proposed`` counts, ``(N, n_runs, n_moves)`` int64 — the unaggregated
    form of what ``adaptation.h5`` stores as rates.

``<prefix>.max_neighbors.h5``
    Opt-in via ``output.save_max_neighbors``, and a no-op for backends with
    no neighbour list.  ``max_neighbor_count`` ``(N, n_runs, n_walkers)``
    int32 against ``bucket_size`` ``(N, n_runs)`` int32 and an ``overflow``
    bool — this is how you size ``backend.max_neighbors_list``.

``<prefix>.re_stats.h5``
    Opt-in via ``output.save_re_stats``, and only meaningful with
    ``inter_re``.  ``n_accepted_per_pair`` and ``n_attempted_per_pair``, both
    ``(N, n_pairs)`` int32 with ``n_pairs = n_runs - 1``, plus the swap
    ``flavor``.  Its cadence follows ``inter_re.re_interval``.  Read with
    :class:`~jaxrens.io.re_stats_log.RELogger`.


``.log``
--------

Plain text: resolver progress, per-``info_interval`` monitor rows, bucket
escalations, adaptation events and the termination reason.  ``jaxrens run``
prints nothing to the console, so this is where you watch a run:

.. code-block:: bash

   tail -f <prefix>.log

.. note::

   The log is written to the **parent** of ``output.working_dir``, not into
   it.  With ``working_dir: ./output`` you get ``./ns.log`` beside the
   ``output/`` directory.  Set ``output.log_level: debug`` for resolver and
   per-iteration detail.
