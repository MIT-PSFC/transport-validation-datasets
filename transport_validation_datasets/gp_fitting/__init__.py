"""Pluggable GP profile fitting: workers, batch io, and cluster dispatch.

A cluster job runs `python -m transport_validation_datasets.gp_fitting.worker_x`
in the venv bootstrap_remote.sh builds, which holds only numpy, scipy and mkgp.
The dispatcher ships this package without dispatcher.py, plus the root __init__.py.
So every module a worker imports may use only the stdlib, numpy, scipy and mkgp,
and nothing from the rest of transport_validation_datasets.

Nothing imported in this file may load numpy.
Python runs it before the worker module,
and the worker pins BLAS threads before its first numpy import.
Loading numpy here would make that pin too late and oversubscribe cluster jobs.
The same holds for the root transport_validation_datasets/__init__.py.
The method subpackages load after the pin and need no such care.
"""
