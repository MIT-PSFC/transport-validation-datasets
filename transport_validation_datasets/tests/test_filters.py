"""Unit tests for the shot checks of filters.py that no workflow test drives."""

import numpy as np
import xarray as xr

from transport_validation_datasets.filters import ohmic_power_sign_reason


def _kept_dataset(power_ohm: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"power_ohm": ("time", power_ohm)},
        coords={"time": np.arange(power_ohm.size) * 1e-3},
    )


class TestOhmicPowerSignReason:
    def test_negative_median_is_rejected(self):
        reason = ohmic_power_sign_reason(
            _kept_dataset(np.array([-3e5, -2e5, 1e5, np.nan]))
        )
        assert reason is not None
        assert "negative" in reason

    def test_positive_clipped_and_all_nan_pass(self):
        assert (
            ohmic_power_sign_reason(_kept_dataset(np.array([3e5, 2e5, -1e5]))) is None
        )
        assert ohmic_power_sign_reason(_kept_dataset(np.zeros(4))) is None
        assert ohmic_power_sign_reason(_kept_dataset(np.full(4, np.nan))) is None
