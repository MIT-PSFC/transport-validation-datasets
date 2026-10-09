"""The IDA worker carries already-fitted profiles onto the fit grid with their errors."""

import numpy as np

from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_OK,
    STATUS_SKIPPED,
    FitBatch,
    ShotFitInput,
    default_fit_bounds,
)
from transport_validation_datasets.gp_fitting.worker_ida import (
    GRADIENT_ERROR_FLOOR,
    GRADIENT_ERROR_FRACTION,
    fit_batch,
)

X_STAR = np.linspace(0.0, 1.6, 81)
# IDA points stop at rho_tor_norm 1.2, inside the fit grid
X_POINTS = np.linspace(0.0, 1.2, 241)
POINT_ERROR = 0.05


def _batch(te_rows: np.ndarray, ne_rows: np.ndarray) -> FitBatch:
    n_rows = te_rows.shape[0]
    x_rows = np.tile(X_POINTS, (n_rows, 1))
    error_rows = np.full(x_rows.shape, POINT_ERROR)
    shot_input = ShotFitInput(
        x=x_rows,
        te_y=te_rows,
        te_err=error_rows,
        ne_y=ne_rows,
        ne_err=error_rows,
        time=np.arange(n_rows, dtype=float),
    )
    return FitBatch(
        shot_inputs={1: shot_input},
        x_star=X_STAR,
        min_points=10,
        scale_per_slice=False,
        bounds=default_fit_bounds(),
        anchors={},
        pedestal_rho_tor_norm=1.0,
        sol_extension="secant",
    )


def test_parabola_carried_with_errors_and_nan_past_the_points():
    te_points = 2.0 * (1.0 - 0.5 * X_POINTS**2)
    ne_points = 1.0 - 0.3 * X_POINTS**2
    batch = _batch(te_points[None, :], ne_points[None, :])

    output = fit_batch(batch)[1]

    inside = X_STAR <= 1.2
    te_expected = 2.0 * (1.0 - 0.5 * X_STAR[inside] ** 2)
    assert output.te_status[0] == STATUS_OK
    assert np.allclose(output.te_fit[0, inside], te_expected, atol=1e-4)
    assert np.isnan(output.te_fit[0, ~inside]).all()
    assert np.allclose(output.te_std[0, inside], POINT_ERROR)
    # Central differences of a parabola are exact, the one-sided ends are not
    interior = (X_STAR > 0.0) & (X_STAR < 1.15)
    te_gradient_expected = -2.0 * X_STAR[interior]
    assert np.allclose(output.te_grad[0, interior], te_gradient_expected, atol=1e-3)
    # The stand-in: a tenth of |gradient|, floored, so the floor binds inside x = 0.5 and the fraction outside
    gradient_error_expected = np.maximum(
        GRADIENT_ERROR_FRACTION * np.abs(output.te_grad[0, interior]),
        GRADIENT_ERROR_FLOOR["te"],
    )
    assert np.allclose(output.te_grad_std[0, interior], gradient_error_expected)
    assert (
        output.te_grad_std[0, interior & (X_STAR < 0.4)] == GRADIENT_ERROR_FLOOR["te"]
    ).all()
    assert (
        output.te_grad_std[0, interior & (X_STAR > 0.6)] > GRADIENT_ERROR_FLOOR["te"]
    ).all()
    assert np.isnan(output.te_grad_std[0, ~inside]).all()


def test_row_with_too_few_points_is_skipped():
    te_rows = np.full((2, X_POINTS.size), np.nan)
    te_rows[0] = 1.0
    te_rows[1, :5] = 1.0
    ne_rows = np.ones((2, X_POINTS.size))
    batch = _batch(te_rows, ne_rows)

    output = fit_batch(batch)[1]

    assert output.te_status[0] == STATUS_OK
    assert output.te_status[1] == STATUS_SKIPPED
    assert np.isnan(output.te_fit[1]).all()
    assert output.ne_status[1] == STATUS_OK
