"""The zk worker's per-slice robustness, without a real GP fit.

The per-variable fit is monkeypatched on a three-slice batch:
one slice raises or comes back non-finite, the others fit,
and the batch must come back with that one slice STATUS_FAILED and the rest STATUS_OK.
"""

from functools import partial

import numpy as np
import pytest

import transport_validation_datasets.gp_fitting.worker_zk as worker_zk
from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_FAILED,
    STATUS_OK,
    FitAnchors,
    FitBatch,
    ShotFitInput,
    default_fit_bounds,
)
from transport_validation_datasets.gp_fitting.worker_base import (
    VariableFit,
    map_slices,
)
from transport_validation_datasets.gp_fitting.zk.gp import ProfileFit

X_CHANNELS = np.linspace(0.1, 0.9, 6)
X_STAR = np.linspace(0.0, 1.2, 13)
# A value no profile reaches, marking the slice that must fail
BAD_MARKER = 999.0
BAD_SLICE = 1
N_SLICES = 3


def _anchors() -> FitAnchors:
    return FitAnchors(
        value=np.array([[1.3, 0.0, 0.05]]), grad=np.array([[1.3, 0.0, 0.5]])
    )


def _batch() -> FitBatch:
    x_rows = np.tile(X_CHANNELS, (N_SLICES, 1))
    y_rows = np.tile(1.0 - X_CHANNELS**2, (N_SLICES, 1))
    y_rows[BAD_SLICE, 0] = BAD_MARKER
    err_rows = np.full(x_rows.shape, 0.05)
    shot_input = ShotFitInput(
        x=x_rows,
        te_y=y_rows,
        te_err=err_rows,
        ne_y=y_rows.copy(),
        ne_err=err_rows,
        time=np.arange(N_SLICES, dtype=float),
    )
    return FitBatch(
        shot_inputs={1: shot_input},
        x_star=X_STAR,
        min_points=3,
        scale_per_slice=False,
        bounds=default_fit_bounds(),
        anchors={"te": _anchors(), "ne": _anchors()},
        pedestal_rho_tor_norm=1.0,
        sol_extension="secant",
    )


def _is_bad(y: np.ndarray) -> bool:
    return bool(np.nanmax(y) >= BAD_MARKER)


def _healthy_fit() -> ProfileFit:
    profile = np.maximum(1.0 - X_STAR**2, 0.0)
    return ProfileFit(
        fit=profile,
        std=np.full_like(profile, 0.05),
        grad=-2.0 * X_STAR,
        grad_std=np.full_like(profile, 0.1),
        hyps=np.array([2.0, 0.5, 0.3, 0.1]),
    )


def _expected_statuses() -> list[int]:
    expected = np.full(N_SLICES, STATUS_OK)
    expected[BAD_SLICE] = STATUS_FAILED
    return expected.tolist()


def test_slice_that_raises_fails_alone_in_a_pool_and_raises_serially(monkeypatch):
    def fit_variable(x, y, err, *args):
        if _is_bad(y):
            raise RuntimeError("mkgp blew up")
        healthy = _healthy_fit()
        return VariableFit(
            healthy.fit,
            healthy.std,
            healthy.grad,
            healthy.grad_std,
            healthy.hyps,
            STATUS_OK,
        )

    monkeypatch.setattr(worker_zk, "_fit_variable", fit_variable)

    # The guarded fit a pool runs, here in the serial loop so the monkeypatch holds
    guarded_fit_slice = partial(worker_zk._fit_slice, guarded=True)
    output = map_slices(guarded_fit_slice, _batch())[1]

    assert output.te_status.tolist() == _expected_statuses()
    assert output.ne_status.tolist() == _expected_statuses()
    assert np.isnan(output.te_fit[BAD_SLICE]).all()
    assert np.isfinite(output.te_fit[[0, 2]]).all()
    # A serial run is the local one, where the error must reach the debugger
    with pytest.raises(RuntimeError, match="mkgp blew up"):
        worker_zk.fit_batch(_batch(), num_workers=1)


def test_non_finite_fit_fails_alone(monkeypatch):
    def clean_channels(x, y, err, bounds, anchors, pedestal_rho, scale_per_slice):
        valid = np.isfinite(y)
        return x[valid], y[valid], err[valid], 1.0, anchors

    def fit_profile(data_x, data_y, *args, **kwargs):
        healthy = _healthy_fit()
        if _is_bad(data_y):
            healthy.fit[:] = np.nan
        return healthy

    monkeypatch.setattr(worker_zk, "clean_channels", clean_channels)
    monkeypatch.setattr(worker_zk, "fit_profile", fit_profile)

    output = worker_zk.fit_batch(_batch(), num_workers=1)[1]

    assert output.te_status.tolist() == _expected_statuses()
    assert output.ne_status.tolist() == _expected_statuses()
    assert np.isnan(output.ne_fit[BAD_SLICE]).all()
    assert np.isfinite(output.ne_fit[[0, 2]]).all()
