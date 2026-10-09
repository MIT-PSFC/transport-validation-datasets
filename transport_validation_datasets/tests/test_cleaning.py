"""The channel screens a device calls: relative dips, shot-long biased channels, and the error inflation where two chord branches disagree."""

import numpy as np

from transport_validation_datasets.cleaning import (
    branch_disagreement_errors,
    drop_broken_channels,
    low_side_channels,
    relative_dips,
)


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


class TestDropBrokenChannels:
    def _rows(self, n_slices=40):
        # 21 channels evenly in rho over a parabolic profile, 2 percent noise on the 10 percent C-Mod error floors
        rng = np.random.default_rng(0)
        rho = np.linspace(0.0, 1.0, 21)
        rho_rows = np.tile(rho, (n_slices, 1))
        profile = 1.0 - 0.8 * rho**2
        noise = 1.0 + 0.02 * rng.standard_normal((n_slices, rho.size))
        value_rows = profile * noise
        error_rows = 0.1 * np.abs(value_rows)
        return rho_rows, value_rows, error_rows

    def test_channel_biased_all_shot_is_dropped_its_neighbours_kept(self):
        rho_rows, value_rows, error_rows = self._rows()
        value_rows[:, 8] *= 1.6

        value_out = drop_broken_channels("te", rho_rows, value_rows, error_rows)

        dropped = np.isnan(value_out).all(axis=0)
        assert np.flatnonzero(dropped).tolist() == [8]

    def test_channel_biased_in_a_minority_of_slices_is_kept(self):
        rho_rows, value_rows, error_rows = self._rows()
        value_rows[:15, 8] *= 1.6

        value_out = drop_broken_channels("te", rho_rows, value_rows, error_rows)

        assert np.isfinite(value_out).all()


def two_branch_slice(inboard_offset: float = 0.0, n_outboard: int = 60):
    # One slice of a chord through the axis at R = 0.85 m, with rho linear in |R - 0.85|.
    # 60 inboard channels, the MAST core spacing, and n_outboard outboard ones, interleaved in rho.
    r_inboard = np.linspace(0.30, 0.845, 60)
    r_outboard = np.linspace(0.852, 1.40, n_outboard)
    r_channel = np.concatenate([r_inboard, r_outboard])[None, :]
    rho = np.abs(r_channel - 0.85) / 0.55
    profile = 1.0 - 0.8 * rho**2
    y = np.where(r_channel < 0.85, profile + inboard_offset, profile)
    err = np.full(y.shape, 0.01)
    return rho, r_channel, y, err


class TestBranchDisagreementErrors:
    def test_offset_branches_get_half_the_offset_in_quadrature(self):
        rho, r_channel, y, err = two_branch_slice(inboard_offset=0.1)
        inboard = low_side_channels(rho, r_channel)

        err_out = branch_disagreement_errors(rho, y, err, inboard)

        assert (inboard == (r_channel < 0.85)).all()
        # The two ends of the overlap have fewer than BRANCH_MIN_CHANNELS estimates in reach
        interior = (rho > 0.05) & (rho < 0.95)
        expected = np.hypot(0.01, 0.05)
        assert np.allclose(err_out[interior], expected, rtol=1e-2)

    def test_sparse_branch_is_inflated_like_the_dense_one(self):
        # 20 outboard channels ~0.05 apart in rho, like the MAST outboard edge,
        # too few for a window of their own branch
        rho, r_channel, y, err = two_branch_slice(inboard_offset=0.1, n_outboard=20)
        inboard = low_side_channels(rho, r_channel)

        err_out = branch_disagreement_errors(rho, y, err, inboard)

        interior_outboard = ~inboard & (rho > 0.05) & (rho < 0.95)
        expected = np.hypot(0.01, 0.05)
        assert np.allclose(err_out[interior_outboard], expected, rtol=1e-2)

    def test_channels_past_the_overlap_keep_their_errors(self):
        # The inboard branch stops at rho 0.7, as past MAX_INBOARD_RHO_TOR_NORM, the outboard one runs on to rho 1
        rho, r_channel, y, err = two_branch_slice(inboard_offset=0.1)
        inboard = low_side_channels(rho, r_channel)
        y[inboard & (rho > 0.7)] = np.nan

        err_out = branch_disagreement_errors(rho, y, err, inboard)

        overlap = ~inboard & (rho > 0.05) & (rho < 0.65)
        past = ~inboard & (rho > 0.8)
        assert np.allclose(err_out[overlap], np.hypot(0.01, 0.05), rtol=1e-2)
        assert np.allclose(err_out[past], 0.01)

    def test_agreeing_branches_keep_their_errors(self):
        rho, r_channel, y, err = two_branch_slice()
        inboard = low_side_channels(rho, r_channel)

        err_out = branch_disagreement_errors(rho, y, err, inboard)

        assert np.allclose(err_out, err, rtol=1e-2)

    def test_one_spike_inflates_nothing(self):
        # The running median outvotes the spike, on its own branch and on the other,
        # and leaves it to the spike screen with its own error
        rho, r_channel, y, err = two_branch_slice()
        i_spike = 90
        y[0, i_spike] += 0.5
        inboard = low_side_channels(rho, r_channel)

        err_out = branch_disagreement_errors(rho, y, err, inboard)

        assert np.allclose(err_out, err, rtol=1e-2)
