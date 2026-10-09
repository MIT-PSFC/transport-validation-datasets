"""Fit-method registry: map a method string to its worker module.

The method string is how a fitting method is selected everywhere:
locally (run_batch_file) and on the cluster,
where the dispatcher composes the job command from worker_module():

    srun python -m {worker_module(method)} in.npz out.npz --num-workers N
"""

import importlib
from pathlib import Path
from types import ModuleType

WORKER_MODULES: dict[str, str] = {
    "zk": "transport_validation_datasets.gp_fitting.worker_zk",
    "akho": "transport_validation_datasets.gp_fitting.worker_akho",
    "ida": "transport_validation_datasets.gp_fitting.worker_ida",
}


def worker_module(method: str) -> str:
    """Get the dotted module path of a fitting method's worker.

    Args:
        method: Fitting method name.

    Returns:
        Dotted module path, e.g.
        "transport_validation_datasets.gp_fitting.worker_zk".

    Raises:
        ValueError: If the method is not registered.
    """
    if method not in WORKER_MODULES:
        raise ValueError(
            f"Unknown fitting method '{method}'. "
            f"Known methods: {sorted(WORKER_MODULES)}"
        )
    return WORKER_MODULES[method]


def load_worker(method: str) -> ModuleType:
    """Import and return a fitting method's worker module.

    Lazy on purpose: importing the registry never pulls numpy, mkgp,
    or an unimplemented method's dependencies.

    Args:
        method: Fitting method name.

    Returns:
        The worker module, providing fit_batch and main.
    """
    return importlib.import_module(worker_module(method))


def run_batch_file(method: str, input_path: Path | str, output_path: Path | str):
    """Fit one staged batch file in this process, one slice at a time, writing its result file.

    Serial and in-process, so a breakpoint in a worker stops there.
    It reads the same staged npz a cluster job would,
    so the fits match a cluster run's as far as the numpy and scipy builds agree
    (bootstrap_remote.sh pins the cluster's versions).

    One difference is numpy keeps the BLAS thread count this process started with,
    where a cluster job pins one.

    Args:
        method: Fitting method name.
        input_path: Batch input npz.
        output_path: Batch result npz to write.
    """
    from transport_validation_datasets.gp_fitting import batch_io

    worker = load_worker(method)
    batch = batch_io.unpack_fit_batch(input_path)
    outputs = worker.fit_batch(batch, num_workers=1)
    batch_io.pack_fit_results(output_path, outputs, batch.x_star)
