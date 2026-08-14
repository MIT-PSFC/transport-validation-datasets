import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, colorize=True)


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
