"""Pluggable GP profile fitting: workers, batch io, and cluster dispatch.

KEEP THIS FILE FREE OF IMPORTS (especially anything that loads numpy). The
worker modules pin BLAS threads via os.environ.setdefault before numpy first
loads, and `python -m ...gp_fitting.worker_x` executes this file first - a
numpy import here (even indirectly, e.g. a convenience re-export) would load
BLAS before the pinning runs and silently oversubscribe cluster jobs.
"""
