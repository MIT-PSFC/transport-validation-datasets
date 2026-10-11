"""Unit tests for the shot checks of filters.py that no workflow test drives."""

import numpy as np
import xarray as xr

from transport_validation_datasets.filters import (
    energy_crash_mask,
    energy_sanity_reason,
    ohmic_power_sign_reason,
)
from transport_validation_datasets.store_schema import INPUT_POWERS

DT = 1e-3
CRASH_WINDOW = 0.05
CRASH_DROP_OVER_PEAK = 0.5


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


def _crash_mask(energy_mhd: np.ndarray) -> np.ndarray:
    mask_before_end = np.ones(energy_mhd.size, dtype=bool)
    return energy_crash_mask(
        energy_mhd, DT, mask_before_end, CRASH_WINDOW, CRASH_DROP_OVER_PEAK
    )


class TestEnergyCrashMask:
    def test_crash_flags_the_window_before_it_and_the_fall_but_not_the_recovery(self):
        # 1 MJ falls to 0.3 MJ over 30 ms from index 500, then recovers to 0.8 MJ by index 700
        energy_mhd = np.ones(1000)
        energy_mhd[500:531] = np.linspace(1.0, 0.3, 31)
        energy_mhd[531:700] = np.linspace(0.3, 0.8, 169)
        energy_mhd[700:] = 0.8

        mask_crash = _crash_mask(energy_mhd)

        # W first sits 0.5 below 1 MJ at index 522, within the 50 ms window from index 472 on,
        # and the fall is cut down to its lowest point at index 530, though from index 509 on less than 0.5 is left to fall
        flagged = np.flatnonzero(mask_crash)
        assert flagged.min() == 472
        assert flagged.max() == 530
        assert flagged.size == 530 - 472 + 1

    def test_sawteeth_slow_decay_and_ramp_up_noise_pass(self):
        time = np.arange(3000) * DT
        # Sawteeth dropping 0.3 of the peak every 100 ms, then recovering with a 20 ms time constant
        energy_sawteeth = np.where((time % 0.1) < DT, 0.7, 1.0 - 0.3 * np.exp(-(time % 0.1) / 0.02))
        # A decay to a fifth over a second, about 0.04 of the peak per 50 ms
        energy_decay = np.where(time < 1.0, 1.0, np.maximum(0.2, 1.0 - 0.8 * (time - 1.0)))
        # The ramp-up's low stored energy dropping to nothing, a small part of the peak, and a NaN gap
        energy_ramp_up = np.where(time < 0.5, 0.05 * (time % 0.05 > 0.02), 1.0)
        energy_ramp_up[2000:2100] = np.nan

        assert not _crash_mask(energy_sawteeth).any()
        assert not _crash_mask(energy_decay).any()
        assert not _crash_mask(energy_ramp_up).any()

    def test_one_high_glitch_cannot_set_the_peak_of_a_crash(self):
        # A 0.6 MJ crash under a 1 MJ plateau holding a single 5 MJ reconstruction glitch
        energy_mhd = np.ones(1000)
        energy_mhd[200] = 5.0
        energy_mhd[600:] = 0.4

        mask_crash = _crash_mask(energy_mhd)

        assert mask_crash[580]
        assert not mask_crash[300]


def _energy_dataset(energy_mhd: np.ndarray, power_ohm: float) -> xr.Dataset:
    powers = {name: ("time", np.zeros(energy_mhd.size)) for name in INPUT_POWERS}
    powers["power_ohm"] = ("time", np.full(energy_mhd.size, power_ohm))
    return xr.Dataset(
        {"energy_mhd": ("time", energy_mhd), **powers},
        coords={"time": 0.5 + np.arange(energy_mhd.size) * DT},
    )


class TestEnergySanityStartGlitches:
    def test_spike_and_held_step_at_the_kept_start_pass(self):
        # 2 kJ with a 3.6 kJ reconstruction spike 2 ms in, against 0.12 MW (TCV 67075)
        energy_spike = np.full(1400, 2e3)
        energy_spike[2] = 3.6e3
        # A held reconstruction stepping 6 to 16 kJ 10 ms in, against 0.4 MW (MAST 23884)
        energy_step = np.where(np.arange(350) < 10, 6e3, 1.6e4)

        assert energy_sanity_reason(_energy_dataset(energy_spike, 1.2e5)) is None
        assert energy_sanity_reason(_energy_dataset(energy_step, 4e5)) is None

    def test_rise_without_heating_record_still_rejected(self):
        # 100 kJ built up over 0.3 s on 0.1 MW of ohmic power, 30 kJ in all
        energy_rise = np.linspace(1e4, 1.1e5, 300)

        reason = energy_sanity_reason(_energy_dataset(energy_rise, 1e5))

        assert reason is not None
        assert "start level" in reason
