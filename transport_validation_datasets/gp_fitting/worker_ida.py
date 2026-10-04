"""DIII-D IDA profiles, the GP fits of integrated data analysis, carried onto the fit grid.

IDA has already fit Te and ne, so nothing is fit here.
Each staged row holds one IDA slice's points, mapped onto rho_tor_norm by the device's prepare_fit_input,
and the profile and its error are interpolated onto x_star.
Inside the IDA points the interpolation holds the innermost value at the axis,
past the outermost point it is NaN.
The gradient is the finite difference on x_star.
IDA gives no point covariance, so the gradient error is a stand-in:
GRADIENT_ERROR_FRACTION of |gradient| with a GRADIENT_ERROR_FLOOR per variable,
the proportion the GP fits of the other devices show at mid radius.
The anchors and bounds of the batch are ignored, they shape GP fits made here.
The workflow refuses this method on any device other than DIII-D (DataWorkflow.fit_methods).

Runs standalone like every worker:
`python -m transport_validation_datasets.gp_fitting.worker_ida input.npz output.npz`

Ships to the cluster with the worker, must adhere to import rules in gp_fitting/__init__.py.
"""

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_OK,
    FitBatch,
    ShotFitOutput,
)
from transport_validation_datasets.gp_fitting.worker_base import run_worker_cli

# The method's own name for its GP fit, in the description of every fitted profile
FIT_DESCRIPTION = "IDA"


# Stand-in gradient error, fraction of |gradient| and the floor per variable [keV and 1e20 m^-3 per unit rho_tor_norm].
# The C-Mod GP fits give grad_err / |grad| of 0.15 (Te) and 0.10 (ne) at mid radius,
# near-axis medians of 0.36 keV and 0.065e20 and edge tenth percentiles of 0.10 keV and 0.11e20.
GRADIENT_ERROR_FRACTION = 0.10
GRADIENT_ERROR_FLOOR = {"te": 0.1, "ne": 0.05}


def stand_in_gradient_error(gradient: np.ndarray, floor: float) -> np.ndarray:
    """A gradient error of GRADIENT_ERROR_FRACTION of |gradient|, at least floor, NaN where the gradient is.

    Args:
        gradient: (..., n_x) the fitted gradient.
        floor: The smallest error [same units as gradient].

    Returns:
        (..., n_x) 1-sigma errors of the gradient.
    """
    gradient_error = np.maximum(GRADIENT_ERROR_FRACTION * np.abs(gradient), floor)
    return np.where(np.isfinite(gradient), gradient_error, np.nan)


def fit_batch(batch: FitBatch, num_workers: int = 1) -> dict[int, ShotFitOutput]:
    """Carry every IDA slice of every shot in the batch onto x_star.

    A variable of a row with fewer than min_points finite points stays STATUS_SKIPPED.

    Args:
        batch: The staged batch.
        num_workers: Ignored, the interpolation is instant.

    Returns:
        Per-shot outputs, row-aligned with the inputs.
    """
    x_star = np.asarray(batch.x_star, dtype=float)
    outputs = {}
    for shot, shot_input in batch.shot_inputs.items():
        shot_output = ShotFitOutput.empty(
            shot_input.time.size, x_star.size, shot_input.time
        )
        for row in range(shot_input.time.size):
            x_row = np.asarray(shot_input.x[row], dtype=float)
            for var in ("te", "ne"):
                y_row = np.asarray(getattr(shot_input, f"{var}_y")[row], dtype=float)
                err_row = np.asarray(
                    getattr(shot_input, f"{var}_err")[row], dtype=float
                )
                mask_valid = (
                    np.isfinite(x_row) & np.isfinite(y_row) & np.isfinite(err_row)
                )
                if int(mask_valid.sum()) < batch.min_points:
                    continue
                order = np.argsort(x_row[mask_valid])
                x_points = x_row[mask_valid][order]
                y_points = y_row[mask_valid][order]
                err_points = err_row[mask_valid][order]
                profile = np.interp(x_star, x_points, y_points, right=np.nan)
                profile_error = np.interp(x_star, x_points, err_points, right=np.nan)
                getattr(shot_output, f"{var}_fit")[row] = profile
                getattr(shot_output, f"{var}_std")[row] = profile_error
                gradient = np.gradient(profile, x_star)
                getattr(shot_output, f"{var}_grad")[row] = gradient
                getattr(shot_output, f"{var}_grad_std")[row] = stand_in_gradient_error(
                    gradient, GRADIENT_ERROR_FLOOR[var]
                )
                getattr(shot_output, f"{var}_status")[row] = STATUS_OK
        outputs[shot] = shot_output
    return outputs


def main(argv: list[str] | None = None):
    """Run the worker CLI: input.npz output.npz [--num-workers N].

    Args:
        argv: Command-line arguments, None uses sys.argv.
    """
    run_worker_cli(fit_batch, prog="worker_ida", argv=argv)


if __name__ == "__main__":
    main()
