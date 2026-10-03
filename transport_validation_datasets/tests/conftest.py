import os
import sys

import pytest
from loguru import logger

from transport_validation_datasets.gp_fitting.registry import WORKER_MODULES
from transport_validation_datasets.workflow import DataWorkflow

logger.remove()
logger.add(sys.stderr, colorize=True)


@pytest.fixture(autouse=True)
def linear_method(monkeypatch):
    # The linear interpolation fit method in tests/linear_worker.py, registered
    # as 'linear' for every test so the workflow stages can run in seconds.
    # Local only: the cluster dispatcher ships gp_fitting/, not tests/.
    # Every device fit with the default GP methods takes it too.
    monkeypatch.setitem(
        WORKER_MODULES, "linear", "transport_validation_datasets.tests.linear_worker"
    )
    gp_and_linear_methods = (*DataWorkflow.fit_methods, "linear")
    monkeypatch.setattr(DataWorkflow, "fit_methods", gp_and_linear_methods)


class _SyncPool:
    """In-process stand-in for multiprocessing.Pool.

    disruption_py's get_shots_data always forks a Pool, even at
    num_processes=1. Under debugpy the forked child is patched by
    pydev_monkey._on_forked_process, hangs trying to attach, and the whole
    debug session dies with no results. Swapping in this shim keeps physics
    methods on the traced main thread so breakpoints fire.
    """

    def __init__(self, processes=None, *args, **kwargs):
        pass

    def imap(self, func, iterable, chunksize=1):
        return map(func, iterable)

    def imap_unordered(self, func, iterable, chunksize=1):
        return map(func, iterable)

    def map(self, func, iterable, chunksize=None):
        return [func(item) for item in iterable]

    def starmap(self, func, iterable, chunksize=None):
        return [func(*item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


if os.environ.get("DISPY_SYNC"):
    import disruption_py.workflow

    disruption_py.workflow.Pool = _SyncPool
    logger.info("DISPY_SYNC set: multiprocessing.Pool swapped for in-process shim")


def pytest_exception_interact(node, call, report):
    print(f"\n=== FAIL {node.nodeid} ===\n{report.longreprtext}", file=sys.stderr)
