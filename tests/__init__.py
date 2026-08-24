"""Test package root.

Present so test modules import under their full dotted path
(``tests.io.test_io``) rather than as top-level modules.  Without it pytest
puts ``tests/`` on ``sys.path`` and ``tests/io/`` shadows the stdlib ``io``
module -- which is why this directory used to be named ``log_io``.  Every
subdirectory already carries an ``__init__.py``; this completes the chain.
"""
