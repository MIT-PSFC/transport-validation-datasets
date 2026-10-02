"""The filter spec every device store shares.

This package applies it to C-Mod and MAST (DataWorkflow.filter_and_plot and shot_rejection_reason),
and POPSIM-Transport-Predictor to DIII-D and TCV (RawFileWorkflow.filter_ds and cull_shot).
"""

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid

from transport_validation_datasets import TIME_COORD
from transport_validation_datasets.machine.generic import (
    centered_boxcar_mean,
    end_of_shot_index,
    greenwald_fraction,
    kept_segments,
)
from transport_validation_datasets.store_schema import (
    DATASET_0D_SIGNALS,
    INPUT_POWERS,
    POWER_SIGNALS,
)

# Width of the centered boxcar applied before the transient thresholds are checked [s].
# It only selects grid times, no stored value is smoothed by it.
TRANSIENT_SMOOTHING_WINDOW = 5e-3

# Margin cut before every grid time that fails a check [s], see slice_filter_mask.
# power_ohm and power_radiated are smoothed non-causally (smoothed_power, DIII-D's sources),
# so they rise ahead of the event that ends a segment.
FAILURE_MARGIN = 20e-3

# How far a shot's stored-energy rise may exceed the input energy put in before it is rejected.
# The 5 percent covers integration error and EFIT noise, see energy_sanity_reason.
ENERGY_SANITY_LEEWAY = 1.05


def filter_inputs(ds: xr.Dataset) -> xr.Dataset:
    """The signals min_filter and max_filter judge, with ip as its magnitude and the derived greenwald_fraction.

    Args:
        ds: One shot's dataset with standardized names.

    Returns:
        The dataset with ip replaced by |ip| and greenwald_fraction added.
    """
    ip_magnitude = abs(ds["ip"])
    fraction = greenwald_fraction(
        ip_magnitude, ds["minor_radius"], ds["n_e_line_average"]
    )
    return ds.assign(ip=ip_magnitude, greenwald_fraction=fraction)


def slice_filter_mask(
    ds: xr.Dataset,
    times: np.ndarray,
    min_filter: dict[str, float],
    max_filter: dict[str, float],
    transient_filter: dict[str, float],
    end_margin: float,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """The grid times of one shot that pass the per-time checks of the filter spec.

    Every check cuts the grid times it fails out as a gap:
    the end of the shot (end_of_shot_index on |ip| and its min_filter threshold),
    a DATASET_0D_SIGNALS signal that is not finite,
    a min_filter signal below its threshold or a max_filter signal above it, judged on filter_inputs,
    and a transient_filter signal above its threshold after a centered TRANSIENT_SMOOTHING_WINDOW boxcar.
    Each grid time that fails a check before the end-of-shot cut also cuts the FAILURE_MARGIN before it,
    so a segment that ends on a failure ends that much earlier, and one that ends at the end-of-shot cut does not.
    Keeping the longest segment of what passes is left to the caller.

    Args:
        ds: One shot's signals on its time dimension alone.
        times: (n_t,) the shot's uniform timebase [s].
        min_filter: {signal: threshold}, ip among them.
        max_filter: {signal: threshold}.
        transient_filter: {signal: threshold} on the smoothed signal.
        end_margin: Margin cut before the end of the plasma [s].

    Returns:
        (mask_valid, mask_transient, end_cut_index):
        the grid times that pass (the failure margins cut), the transients among them,
        and the index of the first grid time the end-of-shot cut removes.
        None when |ip| never reaches its min_filter threshold.
    """
    ds_filter_inputs = filter_inputs(ds)
    ip_magnitude = ds_filter_inputs["ip"].values
    end_cut_index = end_of_shot_index(ip_magnitude, times, min_filter["ip"], end_margin)
    if end_cut_index is None:
        return None
    grid_index = np.arange(times.size)
    mask_before_end = grid_index < end_cut_index
    mask_pass = np.ones(times.size, dtype=bool)

    # A NaN fails every threshold
    with np.errstate(invalid="ignore"):
        for signal in DATASET_0D_SIGNALS:
            mask_finite = np.isfinite(ds[signal].values)
            mask_pass = mask_pass & mask_finite
        for signal, threshold in min_filter.items():
            mask_above = ds_filter_inputs[signal].values >= threshold
            mask_pass = mask_pass & mask_above
        for signal, threshold in max_filter.items():
            mask_below = ds_filter_inputs[signal].values <= threshold
            mask_pass = mask_pass & mask_below

        # Smoothed, so that sporadic noise spikes on their own do not trip it
        grid_steps = np.diff(times)
        dt = float(np.median(grid_steps))
        mask_transient = np.zeros(times.size, dtype=bool)
        for signal, threshold in transient_filter.items():
            smoothed = centered_boxcar_mean(
                ds[signal].values, TRANSIENT_SMOOTHING_WINDOW, dt
            )
            mask_transient = mask_transient | (smoothed > threshold)
    mask_pass = mask_pass & ~mask_transient
    mask_failed = ~mask_pass & mask_before_end
    margin_steps = round(FAILURE_MARGIN / dt)
    mask_in_margin = _before_failures(mask_failed, margin_steps)
    mask_valid = mask_before_end & mask_pass & ~mask_in_margin
    return mask_valid, mask_transient, end_cut_index


def _before_failures(mask_failed: np.ndarray, margin_steps: int) -> np.ndarray:
    """Mark the grid times with a failed one at most margin_steps later.

    Args:
        mask_failed: (n_t,) the grid times that fail a check.
        margin_steps: The margin in grid steps.

    Returns:
        (n_t,) True where a failure falls in (t, t + margin_steps grid steps].
    """
    failures_so_far = np.cumsum(mask_failed)
    grid_index = np.arange(mask_failed.size)
    index_ahead = np.minimum(grid_index + margin_steps, mask_failed.size - 1)
    return failures_so_far[index_ahead] > failures_so_far


def input_power(ds: xr.Dataset) -> xr.DataArray:
    """The power put into the plasma, the sum of INPUT_POWERS [W].

    Each is clipped at 0 and a NaN counts as 0, so a missing record reads as no input.

    Args:
        ds: One shot's dataset with standardized names.

    Returns:
        The input power on the dataset's times.
    """
    power_input = xr.zeros_like(ds["power_ohm"], dtype=float)
    for name in INPUT_POWERS:
        power = ds[name].fillna(0.0).clip(min=0.0)
        power_input = power_input + power
    return power_input


def clip_powers(ds: xr.Dataset) -> xr.Dataset:
    """Clip every power signal at 0, after filtering.

    Source power records dip negative (bolometer baseline drift, ICRF pickup, beam baselines),
    and no heating or radiated power is physically negative.
    Clipped after filtering, so the filters judge the values the device recorded.

    Args:
        ds: One shot's filtered dataset with standardized names, updated in place.

    Returns:
        The same dataset with every power at or above 0, NaN untouched.
    """
    for name in POWER_SIGNALS:
        attrs = ds[name].attrs
        # Adding 0 turns -0.0 into 0.0
        ds[name] = ds[name].clip(min=0.0) + 0.0
        ds[name].attrs = attrs
    return ds


def radiated_fraction_reason(
    ds: xr.Dataset, min_radiated_fraction: float
) -> str | None:
    """Check the radiated power against the input power, a dead bolometer reads far below it.

    The mean power_radiated, clipped at 0, may not fall below min_radiated_fraction of the mean input_power.
    A shot with no input power is never rejected.

    Args:
        ds: One shot's dataset on its kept times.
        min_radiated_fraction: The floor of the ratio of the means.

    Returns:
        Why the shot is rejected, or None if it passes.
    """
    power_input_mean = float(input_power(ds).mean())
    if power_input_mean <= 0.0:
        return None
    power_radiated = ds["power_radiated"].clip(min=0.0)
    power_radiated_mean = float(power_radiated.mean())
    radiated_fraction = power_radiated_mean / power_input_mean
    if radiated_fraction >= min_radiated_fraction:
        return None
    return (
        f"mean power_radiated {1e-3 * power_radiated_mean:.1f} kW "
        f"is {100 * radiated_fraction:.1f} percent "
        f"of the {1e-3 * power_input_mean:.0f} kW mean input power, "
        f"below the {100 * min_radiated_fraction:.1f} percent floor"
    )


def energy_sanity_reason(ds: xr.Dataset) -> str | None:
    """Check the stored energy against the input energy, a missing input power record lets it rise unexplained.

    The energy_mhd rise from the first kept time to its peak
    may not exceed ENERGY_SANITY_LEEWAY times the input energy put in over the same span.
    The input ignores every loss, so a larger rise means a missing or broken power record.
    The rise rather than the peak, so energy stored before the first kept time needs no input.
    The input is integrated across the filter gaps, the heating did not stop while the samples were cut.

    Args:
        ds: One shot's dataset on its kept times, on its time dimension alone.

    Returns:
        Why the shot is rejected, or None if it passes.
    """
    times = np.asarray(ds[TIME_COORD].values, dtype=float)
    energy_mhd = np.asarray(ds["energy_mhd"].values, dtype=float)
    power_input = input_power(ds).values
    mask_valid = np.isfinite(times) & np.isfinite(energy_mhd)
    if mask_valid.sum() < 2:
        return None
    times_valid = times[mask_valid]
    order = np.argsort(times_valid)
    times_sorted = times_valid[order]
    energy_sorted = energy_mhd[mask_valid][order]
    power_sorted = power_input[mask_valid][order]
    peak = int(np.argmax(energy_sorted))
    energy_rise = energy_sorted[peak] - energy_sorted[0]
    energy_input = trapezoid(power_sorted[: peak + 1], times_sorted[: peak + 1])
    if energy_rise <= ENERGY_SANITY_LEEWAY * energy_input:
        return None
    return (
        f"energy_mhd rises {1e-3 * energy_rise:.1f} kJ "
        f"from {times_sorted[0]:.3f} s to {times_sorted[peak]:.3f} s, "
        f"more than the {1e-3 * energy_input:.1f} kJ of input power put in"
    )


def mask_spans(mask: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    """List the runs of a mask as time spans, for the time-trace plots (plot_unprocessed_data).

    Args:
        mask: Mask over times.
        times: The shot's timebase [s].

    Returns:
        (first, last) time [s] of each run of True samples.
    """
    starts, ends = kept_segments(mask)
    return [
        (float(times[start]), float(times[end - 1])) for start, end in zip(starts, ends)
    ]
