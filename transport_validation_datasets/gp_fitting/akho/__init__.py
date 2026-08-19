"""Internals of the akho fitting method (see worker_akho.py).

Keep this file free of imports: it sits on the worker's `python -m` import
path, and the worker pins BLAS threads before numpy first loads (see
gp_fitting/__init__.py).
"""
