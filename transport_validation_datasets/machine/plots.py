from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from transport_validation_datasets.windows import window_membership

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def _valid_range_lines(
    ax, valid_filter: dict[str, dict[str, float]], signal: str, scale: float, color: str
):
    """Dashed horizontal lines at the signal's valid min/max, in plot units."""
    bounds = valid_filter.get(signal, {})
    for bound in ("min", "max"):
        if bound in bounds:
            ax.axhline(bounds[bound] * scale, color=color, linestyle="--", linewidth=1)


def _abs_if_negative(da: xr.DataArray) -> tuple[xr.DataArray, str]:
    """Absolute value and a sign-flipped legend name for signals stored with a negative sign convention.

    Args:
        da: Signal to check.

    Returns:
        The signal (absolute value if it was negative) and its legend name.
    """
    if float(da.median()) < 0:
        return abs(da), f"-{da.name}"
    return da, str(da.name)


def _signal_ylim(signals: list, floor_zero: bool = True) -> tuple[float, float]:
    """Y-limits from the plotted signals, so the dashed filter lines cannot inflate the scale.

    Args:
        signals: Arrays of already-scaled plotted values.
        floor_zero: If True, pin the lower limit to 0. Otherwise pad below the minimum.

    Returns:
        (low, high) y-limits, falling back to (0, 1) when no finite values exist.
    """
    values = [np.asarray(s, dtype=float) for s in signals]
    values = [v for v in values if np.isfinite(v).any()]
    if not values:
        return (0.0, 1.0)
    lo = min(float(np.nanmin(v)) for v in values)
    hi = max(float(np.nanmax(v)) for v in values)
    if floor_zero:
        lo = 0.0
    else:
        lo = lo * 0.9 if lo > 0 else lo * 1.1
    hi = hi * 1.1
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return (0.0, 1.0)
    return (lo, hi)


def plot_unprocessed_data(
    ds: xr.Dataset,
    fig_path: Path,
    title: str,
    valid_filter: dict[str, dict[str, float]],
    transient_filter: dict[str, float],
    end_margin_time: float,
    transient_spans: list[tuple[float, float]] | None = None,
    kept_spans: list[tuple[float, float]] | None = None,
    window_spans: list[tuple[float, float]] | None = None,
):
    """For all the 0D signals in the dataset, plot them over time and save the figure to disk.

    One plot with several subplots, in the following groups:

    1: ip (in MA) and b0 on left axis, energy_mhd (in MJ) on right axis
    - Also has a vertical red line indicating the end margin time as identified by filter_and_plot
    2: n_e_line_average (in 10^20 m^-3) on left axis, beta_n (unitless) on right axis
    - Also has green dots at the bottom for each time index where the profiles are not NaN
    3: p_oh, p_rad, p_ic, p_lh, p_nbi (all in MW)
    4: minor_radius and major_radius (both in m) on left axis, kappa, tritop, tribot (unitless) on right axis

    For each signal, include dashed lines to indicate its valid range (from valid_filter)

    Args:
        ds: Dataset with standardized signal names for one shot.
        fig_path: Where to save the figure. Parent directories are created if needed.
        title: Figure title.
        valid_filter: Valid ranges per signal, drawn as dashed lines.
        transient_filter: Transient thresholds per signal, drawn as dashed lines
            on the power subplot.
        end_margin_time: Time of the end margin cutoff [s].
        transient_spans: (start, end) time intervals filter_and_plot cut out as transients,
            shaded as red vertical bars on each subplot.
        kept_spans: (start, end) time intervals kept by filter_and_plot,
            shaded as green vertical bars on each subplot.
        window_spans: (start, end) time windows the shotlist asked for, shaded
            as blue vertical bars on each subplot.
    """
    if "shot" in ds.dims:
        ds = ds.isel(shot=0)
    time = ds["time"]

    fig, axes = plt.subplots(4, 1, figsize=(22.4, 16), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
    all_axes = list(axes)

    if kept_spans is not None:
        for ax in axes:
            for span_start, span_end in kept_spans:
                ax.axvspan(span_start, span_end, color="green", alpha=0.2, linewidth=0)
    if transient_spans is not None:
        for ax in axes:
            for span_start, span_end in transient_spans:
                ax.axvspan(span_start, span_end, color="red", alpha=0.3, linewidth=0)
    if window_spans is not None:
        for ax in axes:
            for span_start, span_end in window_spans:
                ax.axvspan(
                    span_start, span_end, color="tab:blue", alpha=0.25, linewidth=0
                )

    # ip and b0 on the left y axis, energy_mhd on the right y axis
    ax_ip = axes[0]
    ip, ip_label = _abs_if_negative(ds["ip"])
    ip_ma = ip / 1e6
    ax_ip.plot(time, ip_ma, label=f"{ip_label} [MA]", color="cyan")
    _valid_range_lines(ax_ip, valid_filter, "ip", 1e-6, "cyan")
    left_signals = [ip_ma]
    if "b0" in ds:
        b0, b0_label = _abs_if_negative(ds["b0"])
        ax_ip.plot(time, b0, label=f"{b0_label} [T]", color="magenta")
        _valid_range_lines(ax_ip, valid_filter, "b0", 1.0, "magenta")
        left_signals.append(b0)
    ax_ip.set_ylabel("Ip [MA] / B0 [T]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_ip.set_ylim(_signal_ylim(left_signals))
    ax_ip.axvline(end_margin_time, color="red")
    ax_ip.legend(
        fontsize=LEGEND_FONTSIZE,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        loc="upper left",
    )
    ax_energy = ax_ip.twinx()
    energy_signals = []
    if "energy_mhd" in ds:
        energy_mhd_MJ = ds["energy_mhd"] / 1e6
        ax_energy.plot(time, energy_mhd_MJ, label="energy_mhd [MJ]", color="red")
        _valid_range_lines(ax_energy, valid_filter, "energy_mhd", 1e-6, "red")
        energy_signals.append(energy_mhd_MJ)
    ax_energy.set_ylabel("energy_mhd [MJ]", fontsize=LABEL_FONTSIZE, color="red")
    ax_energy.set_ylim(_signal_ylim(energy_signals))
    ax_energy.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
    all_axes.append(ax_energy)

    # line average density on the left y axis, normalized beta on the right y axis
    ax_ne = axes[1]
    ne_signals = []
    if "n_e_line_average" in ds:
        ne20 = ds["n_e_line_average"] / 1e20
        ax_ne.plot(time, ne20, label="n_e_line_average", color="white")
        _valid_range_lines(ax_ne, valid_filter, "n_e_line_average", 1e-20, "white")
        ne_signals.append(ne20)
    ax_ne.set_ylabel("n_e [10^20 m^-3]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_ne.set_ylim(_signal_ylim(ne_signals))

    # Dots at 0 for time indices where the TS profiles have data
    # No ts_channel dim means the TS retrieval failed and the column is all NaN
    if "ts_channel_n_e" in ds and "ts_channel" in ds["ts_channel_n_e"].dims:
        fresh_profiles = ds["ts_channel_n_e"].notnull().any(dim="ts_channel")
        ax_ne.plot(
            time,
            np.where(fresh_profiles, 0, np.nan),
            color="green",
            marker="o",
            linestyle="None",
        )

    ax_beta = ax_ne.twinx()
    beta_signals = []
    if "beta_tor_norm" in ds:
        ax_beta.plot(time, ds["beta_tor_norm"], label="beta_tor_norm", color="magenta")
        _valid_range_lines(ax_beta, valid_filter, "beta_tor_norm", 1.0, "magenta")
        beta_signals.append(ds["beta_tor_norm"])
    ax_beta.set_ylabel("Normalized Beta", fontsize=LABEL_FONTSIZE, color="magenta")
    ax_beta.set_ylim(_signal_ylim(beta_signals))
    ax_beta.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
    all_axes.append(ax_beta)

    # Power sources and sinks, with transient thresholds as dashed lines
    ax_power = axes[2]
    power_colors = {
        "power_ohm": "orange",
        "power_radiated": "red",
        "power_nbi": "cyan",
        "power_lh": "yellow",
        "power_ic": "magenta",
    }
    power_signals = []
    for signal, color in power_colors.items():
        if signal in ds:
            power_mw = ds[signal] / 1e6
            ax_power.plot(time, power_mw, label=f"{signal} [MW]", color=color)
            _valid_range_lines(ax_power, valid_filter, signal, 1e-6, color)
            power_signals.append(power_mw)
    for signal, threshold in transient_filter.items():
        ax_power.axhline(
            threshold / 1e6,
            color=power_colors.get(signal, "white"),
            linestyle="--",
            linewidth=1,
        )
    ax_power.set_ylabel("Power [MW]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_power.set_ylim(_signal_ylim(power_signals))
    ax_power.legend(
        fontsize=LEGEND_FONTSIZE,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        loc="upper left",
    )

    # Radii on the left y axis, shaping on the right y axis
    ax_radius = axes[3]
    radius_colors = {
        "minor_radius": "red",
        "geometric_axis_r": "cyan",
    }
    radius_signals = []
    for signal, color in radius_colors.items():
        if signal in ds:
            ax_radius.plot(time, ds[signal], label=signal, color=color)
            _valid_range_lines(ax_radius, valid_filter, signal, 1.0, color)
            radius_signals.append(ds[signal])
    ax_radius.set_ylabel("Radius [m]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_radius.set_ylim(_signal_ylim(radius_signals, floor_zero=False))
    ax_radius.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_radius.legend(
        fontsize=LEGEND_FONTSIZE,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        loc="upper left",
    )
    ax_shape = ax_radius.twinx()
    shape_colors = {
        "elongation": "yellow",
        "triangularity_upper": "lime",
        "triangularity_lower": "green",
    }
    shape_signals = []
    for signal, color in shape_colors.items():
        if signal in ds:
            ax_shape.plot(time, ds[signal], label=signal, color=color)
            _valid_range_lines(ax_shape, valid_filter, signal, 1.0, color)
            shape_signals.append(ds[signal])
    ax_shape.set_ylabel("Shaping", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_shape.set_ylim(_signal_ylim(shape_signals, floor_zero=False))
    ax_shape.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
    ax_shape.legend(
        fontsize=LEGEND_FONTSIZE,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        loc="upper right",
    )
    all_axes.append(ax_shape)

    for ax in axes:
        ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
    for ax in all_axes:
        ax.set_facecolor(FACE_COLOR)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
        try:
            for text in ax.get_legend().get_texts():
                text.set_color(TEXT_COLOR)
        except AttributeError:
            pass

    fig.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def _panel_ylim(
    fit_mean: np.ndarray, data_y: np.ndarray, fallback: float, pad: float = 1.05
) -> float:
    """Compute the top y-limit for a TS fit panel.

    A little over the largest GP fit mean or channel reading of the plotted slices,
    computed across the whole shot so every page shares the same axis.
    The readings count so a fit that passes under its data shows as such.
    They are the staged ones, already through the cleaning screens,
    which in tuning iteration 7 raised the top over the fit's by at most 1.85x.

    Args:
        fit_mean: (n_t, n_x) fitted profiles of the plotted slices.
        data_y: (n_t, n_ch) channel readings of the plotted slices.
        fallback: Limit to use when both are all-NaN or non-positive.
        pad: Multiplicative headroom above the largest value.

    Returns:
        The top y-limit.
    """
    values = np.concatenate([np.ravel(fit_mean), np.ravel(data_y)])
    hi = float(np.nanmax(values)) if np.isfinite(values).any() else np.nan
    if not np.isfinite(hi) or hi <= 0:
        return fallback
    return hi * pad


# Staged sample times are float32, so they match the unprocessed grid times only to rounding
_SAMPLE_TIME_ATOL = 1.0e-6


def _plot_dropped_readings(
    ax, rho: np.ndarray, y: np.ndarray, err: np.ndarray, ylim_top: float
):
    """Plot readings the device dropped from the fit, in red.

    The y-axis follows the fits and the fitted readings,
    so a dropped reading above it is marked by a red arrow at the top edge.

    Args:
        ax: Axes to plot on.
        rho: rho_tor_norm of the readings.
        y: The readings, shaped like rho.
        err: Their errors, shaped like rho.
        ylim_top: The panel's top y-limit.
    """
    valid = np.isfinite(rho) & np.isfinite(y)
    if not valid.any():
        return
    ax.errorbar(
        rho[valid],
        y[valid],
        yerr=err[valid],
        fmt="o",
        ms=4,
        color="tab:red",
        label="dropped channel",
        zorder=3,
    )
    off_scale = valid & (y > ylim_top)
    n_off_scale = int(off_scale.sum())
    if n_off_scale:
        arrow_y = np.full(n_off_scale, 0.97 * ylim_top)
        ax.plot(rho[off_scale], arrow_y, "^", ms=7, color="tab:red", zorder=4)


# Panel labels for the TS fit diagnostic, keyed by the fit_output variable
# prefix: (prefix, profile label, gradient label, fallback y-limit).
_TS_FIT_PANELS = (
    ("te", "Te [keV]", "dTe/drho_tor_norm [keV]", 5.0),
    ("ne", "ne [1e20 m^-3]", "dne/drho_tor_norm [1e20 m^-3]", 1.8),
)


def _plot_fit_band(ax, x: np.ndarray, y: np.ndarray, err: np.ndarray, label: str):
    """Plot a fit mean plus its +-1 sigma band over the finite part of the grid.

    Args:
        ax: Axes to plot on.
        x: rho_tor_norm grid.
        y: Fit mean.
        err: 1-sigma band half-width.
        label: Legend label of the mean line.
    """
    valid = np.isfinite(y)
    if not valid.any():
        return
    ax.plot(x[valid], y[valid], color="black", label=label)
    ax.fill_between(
        x[valid],
        (y - err)[valid],
        (y + err)[valid],
        color="black",
        alpha=0.2,
        label="GP +-1 sigma",
    )


def _style_ts_panel(ax, ylabel: str, title: str):
    """Apply the shared TS fit panel styling.

    Args:
        ax: Axes to style.
        ylabel: Y-axis label.
        title: Panel title.
    """
    ax.set_xlabel("rho_tor_norm")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def plot_ts_fits(
    pdf_path: Path | str,
    shot: int,
    ts_time: np.ndarray,
    rho_tor_norm_ch: np.ndarray,
    channel_data: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_output,
    rho_tor_norm_fit: np.ndarray,
    channel_groups: list[tuple[np.ndarray, str, str]] | None = None,
    max_pages: int | None = None,
    window_bounds: np.ndarray | None = None,
    dropped_readings: tuple | None = None,
) -> int:
    """Save a PDF comparing the GP fits to the raw TS measurements of one shot.

    One page per sampled measurement time, 2x2 panels: Te (top left) and ne
    (top right) with the channel data, GP fit mean and +-1 sigma predictive
    band, then the GP gradients d/drho_tor_norm with their +-1 sigma bands below.
    Fitted hyperparameters are annotated on the profile panels when the
    method provides them.

    Args:
        pdf_path: Destination PDF path.
        shot: Shot number, for the page titles.
        ts_time: (n_t,) measurement times [s].
        rho_tor_norm_ch: (n_t, n_ch) channel rho_tor_norm locations, NaN where invalid.
        channel_data: {"te": (y, err), "ne": (y, err)}, each (n_t, n_ch), in
            the same units the fit consumed (Te [keV], ne [1e20 m^-3]).
        fit_output: ShotFitOutput whose rows align with ts_time.
        rho_tor_norm_fit: (n_x,) rho_tor_norm grid the fits were predicted on.
        channel_groups: Optional (mask, color, label) triples to split the
            channels by diagnostic; one blue "raw TS" group when None. A mask
            is (n_ch,) for a fixed split (C-Mod core vs edge Thomson) or
            (n_t, n_ch) when the split varies per slice.
        max_pages: Evenly sample the fitted slices down to at most this many
            pages. None plots every fitted slice.
        window_bounds: (n_t, 2) start and end [s] of the time window each
            row pools, for window-averaged fits. Titles the page with the
            window instead of a slice time. None for per-sample fits.
        dropped_readings: (times, rho_tor_norm, {var: (y, err)}) of the readings the device
            drops from every fit (DataWorkflow.fit_plot_dropped_readings), drawn in red.
            A page shows those of its own sample, or of every sample in its window.

    Returns:
        The number of pages written; a shot with no fitted slice writes an
        empty PDF.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Only page over slices where the fit produced a real profile. Times with
    # too few channels (or culled fits) are all-NaN and the panels would come
    # out blank.
    has_fit = np.isfinite(fit_output.te_fit).any(axis=-1) | np.isfinite(
        fit_output.ne_fit
    ).any(axis=-1)
    live = np.flatnonzero(has_fit)
    if max_pages is not None and live.size > max_pages:
        # Evenly spread samples rather than a stride, which overshoots
        # max_pages whenever the fitted count is not a multiple of it
        live = live[
            np.unique(np.linspace(0, live.size - 1, max_pages).round()).astype(int)
        ]

    n_ch = rho_tor_norm_ch.shape[1]
    groups = (
        channel_groups
        if channel_groups is not None
        else [(np.ones(n_ch, dtype=bool), "tab:blue", "raw TS")]
    )
    ylims = {}
    for var, _, _, fallback in _TS_FIT_PANELS:
        fit_live = getattr(fit_output, f"{var}_fit")[live]
        data_live = channel_data[var][0][live]
        ylims[var] = _panel_ylim(fit_live, data_live, fallback=fallback)

    with PdfPages(pdf_path) as pdf:
        for i_time in live:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            if window_bounds is None:
                title = f"shot {shot}  t={ts_time[i_time]:.3f} s"
            else:
                start, end = window_bounds[i_time]
                title = f"shot {shot}  t={start:.3f}-{end:.3f} s (window average)"
            if dropped_readings is not None:
                dropped_time, dropped_rho, dropped_by_var = dropped_readings
                if window_bounds is None:
                    on_page = np.isclose(
                        dropped_time, ts_time[i_time], rtol=0.0, atol=_SAMPLE_TIME_ATOL
                    )
                else:
                    page_window = window_bounds[i_time : i_time + 1]
                    on_page = window_membership(dropped_time, page_window)[:, 0]
            for i_var, (var, label, grad_label, _) in enumerate(_TS_FIT_PANELS):
                data_y, err_y = (arr[i_time, :] for arr in channel_data[var])
                rho_tor_norm_at_t = rho_tor_norm_ch[i_time, :]

                ax = axes[0, i_var]
                for mask, color, name in groups:
                    mask_at_t = mask[i_time, :] if mask.ndim == 2 else mask
                    valid = (
                        mask_at_t
                        & np.isfinite(rho_tor_norm_at_t)
                        & np.isfinite(data_y)
                        & np.isfinite(err_y)
                    )
                    if valid.any():
                        ax.errorbar(
                            rho_tor_norm_at_t[valid],
                            data_y[valid],
                            yerr=err_y[valid],
                            fmt="o",
                            ms=4,
                            color=color,
                            label=name,
                            zorder=3,
                        )
                if dropped_readings is not None and var in dropped_by_var:
                    dropped_y, dropped_err = dropped_by_var[var]
                    _plot_dropped_readings(
                        ax,
                        dropped_rho[on_page],
                        dropped_y[on_page],
                        dropped_err[on_page],
                        ylims[var],
                    )
                _plot_fit_band(
                    ax,
                    rho_tor_norm_fit,
                    getattr(fit_output, f"{var}_fit")[i_time, :],
                    getattr(fit_output, f"{var}_std")[i_time, :],
                    "GP fit",
                )
                ax.set_ylim(bottom=0, top=ylims[var])
                _style_ts_panel(ax, label, title)
                # The annotation layout is the zk method's (4 hyperparameters),
                # other methods' diagnostics are skipped here.
                hyps_all = getattr(fit_output, f"{var}_hyps")
                if (
                    hyps_all is not None
                    and hyps_all.shape[1] == 4
                    and np.isfinite(hyps_all[i_time]).all()
                ):
                    var_h, l1, l2, lw = hyps_all[i_time]
                    ax.text(
                        0.98,
                        0.98,
                        f"var={var_h:.2f}  l1={l1:.2f}\nl2={l2:.2f}  lw={lw:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7,
                        family="monospace",
                    )

                ax = axes[1, i_var]
                _plot_fit_band(
                    ax,
                    rho_tor_norm_fit,
                    getattr(fit_output, f"{var}_grad")[i_time, :],
                    getattr(fit_output, f"{var}_grad_std")[i_time, :],
                    "GP gradient",
                )
                ax.axhline(0.0, color="gray", lw=0.8, alpha=0.5)
                _style_ts_panel(ax, grad_label, title)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return len(live)
