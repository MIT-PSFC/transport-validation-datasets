"""Skeleton for the akho GP profile fitting method.

Interface-conforming placeholder: staging, dispatch, and the CLI shape all
work end to end, but fit_batch raises until the method is implemented.
"""

import os

# Pin BLAS threads before numpy loads so slice-level multiprocessing does not
# oversubscribe cores. Only effective when this module is the program entry
# point (the cluster `python -m` path); see worker_zk.py for the full
# rationale.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from transport_validation_datasets.gp_fitting.batch_io import (  # noqa: E402
    FitBatch,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.worker_base import (  # noqa: E402
    run_worker_cli,
)


def fit_batch(batch: FitBatch, num_workers: int = 1) -> dict[int, ShotFitOutput]:
    """Fit every (shot, time slice) in the batch with the akho method.

    Args:
        batch: Staged batch inputs (see batch_io.FitBatch).
        num_workers: Slice-level worker processes.

    Returns:
        Fitted profiles keyed by shot number.

    Raises:
        NotImplementedError: Always; the akho method is not implemented yet.
    """
    raise NotImplementedError(
        "The akho fitting method is not implemented. Use method='zk'."
    )


def main(argv: list[str] | None = None):
    """Run the akho worker CLI: input.npz output.npz --num-workers N.

    Args:
        argv: Command-line arguments; None uses sys.argv.
    """
    run_worker_cli(fit_batch, prog="worker_akho", argv=argv)


if __name__ == "__main__":
    main()
