"""The relative dip screen: a dead channel inside a pedestal is dropped, real shapes are not."""

import numpy as np

from transport_validation_datasets.cleaning import relative_dips


def _pedestal_row(rho):
    """A monotone Te-like profile with a tanh pedestal at rho 1."""
    return 2.0 * (1.0 - rho**2) * 0.5 * (1.0 - np.tanh((rho - 1.0) / 0.03)) + 0.02


class TestRelativeDips:
    def test_dead_channel_in_pedestal_is_dropped(self):
        rho = np.linspace(0.05, 1.05, 21)
        y = _pedestal_row(rho)
        y[16] = 0.016  # rho 0.85, neighbours read ~0.5
        dips = relative_dips(rho[None, :], y[None, :])
        assert dips[0, 16]
        assert dips.sum() == 1

    def test_dead_axis_channel_is_dropped(self):
        rho = np.linspace(0.05, 1.05, 21)
        y = _pedestal_row(rho)
        y[0] = 0.15
        dips = relative_dips(rho[None, :], y[None, :])
        assert dips[0, 0]
        assert dips.sum() == 1

    def test_steep_pedestal_and_hollow_profile_survive(self):
        rho = np.linspace(0.05, 1.05, 21)
        pedestal = _pedestal_row(rho)
        hollow = 1.0 + 0.4 * np.exp(-((rho - 0.3) ** 2) / 0.02) - 0.9 * rho**4
        rows = np.stack([pedestal, hollow])
        dips = relative_dips(np.stack([rho, rho]), rows)
        assert not dips.any()

    def test_pedestal_foot_is_not_judged(self):
        rho = np.linspace(0.05, 1.05, 21)
        y = _pedestal_row(rho)
        y[19] = 0.005  # rho 1.0, between a pedestal channel and a SOL channel
        dips = relative_dips(rho[None, :], y[None, :])
        assert not dips.any()

    def test_nan_channels_are_skipped(self):
        rho = np.linspace(0.05, 1.05, 21)
        y = _pedestal_row(rho)
        y[15] = np.nan
        y[16] = 0.016
        dips = relative_dips(rho[None, :], y[None, :])
        assert dips[0, 16]
        assert not dips[0, 15]
