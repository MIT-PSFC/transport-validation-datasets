"""Checks for worker_akho.py's optional coordinate-substitution step
(_resolve_fit_coordinate) -- see that module's docstring.

Requires the akho method's own dependencies (scipy.optimize, mkgp) to import
worker_akho.py at all, same as any other test touching this method.
"""

import numpy as np
import pytest

from transport_validation_datasets.gp_fitting.batch_io import FitBounds
from transport_validation_datasets.gp_fitting.worker_akho import (
    _ALT_COORDINATES,
    _resolve_fit_coordinate,
)
from transport_validation_datasets.gp_fitting.worker_base import SliceTask
from transport_validation_datasets.workflow import DeviceSettings, _fit_anchors

_DEFAULT_BOUNDS = FitBounds()
_ANCHORS = _fit_anchors(DeviceSettings())


def _task(fit_coordinate="rho", psi_norm=None, qpsi=None) -> SliceTask:
    return SliceTask(
        shot=1,
        i_time=0,
        x=np.array([0.1, 0.5, 0.9]),
        te_y=np.array([1.0, 2.0, 3.0]),
        te_err=np.array([0.1, 0.1, 0.1]),
        ne_y=np.array([1.0, 2.0, 3.0]),
        ne_err=np.array([0.1, 0.1, 0.1]),
        x_star=np.linspace(0, 1, 5),
        min_points=3,
        scale_per_slice=True,
        te_bounds=_DEFAULT_BOUNDS,
        ne_bounds=_DEFAULT_BOUNDS,
        te_anchors=_ANCHORS["te"],
        ne_anchors=_ANCHORS["ne"],
        psi_norm=psi_norm,
        qpsi=qpsi,
        fit_coordinate=fit_coordinate,
    )


def test_rho_is_a_no_op():
    task = _task(fit_coordinate="rho")
    np.testing.assert_array_equal(_resolve_fit_coordinate(task), task.x)


def test_psi_norm_substitution():
    psi_norm = np.array([0.02, 0.3, 0.9])
    task = _task(fit_coordinate="psi_norm", psi_norm=psi_norm)
    np.testing.assert_allclose(_resolve_fit_coordinate(task), psi_norm)


def test_sqrt_psi_norm_substitution():
    psi_norm = np.array([0.04, 0.25, 0.81])
    task = _task(fit_coordinate="sqrt_psi_norm", psi_norm=psi_norm)
    np.testing.assert_allclose(_resolve_fit_coordinate(task), np.sqrt(psi_norm))


def test_phi_norm_substitution_uses_qpsi():
    psi_norm = np.array([0.0, 0.5, 1.0])
    qpsi = np.linspace(1.0, 4.0, 65)
    task = _task(fit_coordinate="phi_norm", psi_norm=psi_norm, qpsi=qpsi)
    result = _resolve_fit_coordinate(task)
    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert result[-1] == pytest.approx(1.0, abs=1e-6)
    assert result[0] < result[1] < result[2]


def test_missing_psi_norm_raises():
    task = _task(fit_coordinate="psi_norm", psi_norm=None)
    with pytest.raises(ValueError, match="requires psi_norm"):
        _resolve_fit_coordinate(task)


def test_unknown_coordinate_raises():
    task = _task(fit_coordinate="banana", psi_norm=np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="Unknown fit_coordinate"):
        _resolve_fit_coordinate(task)


def test_all_alt_coordinates_resolve_without_error():
    psi_norm = np.array([0.02, 0.3, 0.9])
    qpsi = np.linspace(1.0, 4.0, 65)
    for coord in _ALT_COORDINATES:
        task = _task(fit_coordinate=coord, psi_norm=psi_norm, qpsi=qpsi)
        result = _resolve_fit_coordinate(task)
        assert result.shape == psi_norm.shape


def test_synthetic_anchor_channels_keep_nominal_positions():
    """A finite-x channel with NaN psi_norm (a synthetic SOL anchor, see
    cmod_dataset._append_sol_anchor_points) keeps its nominal x in every
    substituted coordinate -- the Te pedestal gate identifies the anchors by
    those exact positions."""
    task = SliceTask(
        shot=1,
        i_time=0,
        x=np.array([0.1, 0.5, 0.9, 1.05, 1.08]),
        te_y=np.array([1.0, 2.0, 3.0, 0.04, 0.03]),
        te_err=np.full(5, 0.1),
        ne_y=np.array([1.0, 2.0, 3.0, 0.5, 0.3]),
        ne_err=np.full(5, 0.1),
        x_star=np.linspace(0, 1, 5),
        min_points=3,
        scale_per_slice=True,
        te_bounds=_DEFAULT_BOUNDS,
        ne_bounds=_DEFAULT_BOUNDS,
        te_anchors=_ANCHORS["te"],
        ne_anchors=_ANCHORS["ne"],
        psi_norm=np.array([0.02, 0.3, 0.85, np.nan, np.nan]),
        qpsi=np.linspace(1.0, 4.0, 65),
        fit_coordinate="sqrt_phi_norm",
    )
    result = _resolve_fit_coordinate(task)
    assert np.all(np.isfinite(result))
    assert result[3] == pytest.approx(1.05)
    assert result[4] == pytest.approx(1.08)
    # The real channels really were substituted, not passed through.
    assert not np.allclose(result[:3], task.x[:3])


def test_sol_channels_beyond_lcfs_stay_distinct():
    """Real SOL channels (psi_norm > 1) must not collapse onto the LCFS in
    the phi_norm pair -- the linear SOL extension keeps them ordered."""
    psi_norm = np.array([0.9, 1.0, 1.03, 1.08])
    qpsi = np.linspace(1.0, 4.0, 65)
    task = SliceTask(
        shot=1,
        i_time=0,
        x=np.array([0.9, 1.0, 1.02, 1.06]),
        te_y=np.full(4, 1.0),
        te_err=np.full(4, 0.1),
        ne_y=np.full(4, 1.0),
        ne_err=np.full(4, 0.1),
        x_star=np.linspace(0, 1, 5),
        min_points=3,
        scale_per_slice=True,
        te_bounds=_DEFAULT_BOUNDS,
        ne_bounds=_DEFAULT_BOUNDS,
        te_anchors=_ANCHORS["te"],
        ne_anchors=_ANCHORS["ne"],
        psi_norm=psi_norm,
        qpsi=qpsi,
        fit_coordinate="sqrt_phi_norm",
    )
    result = _resolve_fit_coordinate(task)
    assert result[1] == pytest.approx(1.0, abs=1e-6)
    assert result[1] < result[2] < result[3]
