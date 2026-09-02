"""A stand-in fit method for the workflow tests: linear interpolation.

The tests register this module under the method name "linear"
(gp_fitting.registry.WORKER_MODULES), which gives the workflow a fit method
that is instant and exactly predictable. Each row's finite (x, y) points are
sorted by x and interpolated linearly onto x_star, with a constant error and
a finite-difference gradient. A row with fewer than min_points finite points
stays STATUS_SKIPPED. Outside the span of a row's points np.interp holds the
end values.
"""

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_OK,
    FitBatch,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.worker_base import run_worker_cli

# The constant 1-sigma error every fitted point and gradient gets.
FIT_STD = 0.05


def fit_batch(batch: FitBatch, num_workers: int = 1) -> dict[int, ShotFitOutput]:
    """Interpolate every row of every shot in the batch onto x_star.

    Args:
        batch: The staged batch.
        num_workers: Ignored, the interpolation is instant.

    Returns:
        Per-shot outputs, row-aligned with the inputs.
    """
    x_star = np.asarray(batch.x_star, dtype=float)
    outputs = {}
    for shot, si in batch.shot_inputs.items():
        so = ShotFitOutput.empty(si.time.size, x_star.size, si.time)
        for row in range(si.time.size):
            x = np.asarray(si.x[row], dtype=float)
            for var in ("te", "ne"):
                y = np.asarray(getattr(si, f"{var}_y")[row], dtype=float)
                err = np.asarray(getattr(si, f"{var}_err")[row], dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
                if int(ok.sum()) < batch.min_points:
                    continue
                order = np.argsort(x[ok])
                fit = np.interp(x_star, x[ok][order], y[ok][order])
                getattr(so, f"{var}_fit")[row] = fit
                getattr(so, f"{var}_std")[row] = FIT_STD
                getattr(so, f"{var}_grad")[row] = np.gradient(fit, x_star)
                getattr(so, f"{var}_grad_std")[row] = FIT_STD
                getattr(so, f"{var}_status")[row] = STATUS_OK
        outputs[shot] = so
    return outputs


def main(argv: list[str] | None = None):
    """Run the worker CLI: input.npz output.npz [--num-workers N].

    Args:
        argv: Command-line arguments; None uses sys.argv.
    """
    run_worker_cli(fit_batch, prog="linear_worker", argv=argv)


if __name__ == "__main__":
    main()
