from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
    transient_margin_time: float | None = None,
    kept_spans: list[tuple[float, float]] | None = None,
):
    """For all the 0D signals in the dataset, plot them over time and save the figure to disk.

    One plot with several subplots, in the following groups:

    1: ip (in MA) and b0 on left axis, wmhd (in MJ) on right axis
    - Also has a vertical red line indicating the end margin time as identified by filter_and_plot
    - And a vertical yellow line indicating the cutoff before a transient event, as identified by filter_and_plot (if provided)
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
        transient_margin_time: Time of the transient cutoff [s], if one was found.
        kept_spans: (start, end) time intervals kept by filter_and_plot, shaded
            as green vertical bars on each subplot.
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

    # ip and b0 on the left y axis, wmhd on the right y axis
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
    if transient_margin_time is not None:
        ax_ip.axvline(transient_margin_time, color="yellow")
    ax_ip.legend(
        fontsize=LEGEND_FONTSIZE,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        loc="upper left",
    )
    ax_wmhd = ax_ip.twinx()
    wmhd_signals = []
    if "energy_mhd" in ds:
        wmhd = ds["energy_mhd"] / 1e6
        ax_wmhd.plot(time, wmhd, label="wmhd [MJ]", color="red")
        _valid_range_lines(ax_wmhd, valid_filter, "energy_mhd", 1e-6, "red")
        wmhd_signals.append(wmhd)
    ax_wmhd.set_ylabel("Wmhd [MJ]", fontsize=LABEL_FONTSIZE, color="red")
    ax_wmhd.set_ylim(_signal_ylim(wmhd_signals))
    ax_wmhd.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
    all_axes.append(ax_wmhd)

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


def _fit_mean_ylim(fit_mean: np.ndarray, fallback: float, pad: float = 1.05) -> float:
    """Compute the top y-limit for a TS fit panel.

    A little over the largest GP fit mean, computed across the whole shot so
    every page shares the same axis.

    Args:
        fit_mean: (n_t, n_x) fitted profiles of the shot.
        fallback: Limit to use when the fit is all-NaN or non-positive.
        pad: Multiplicative headroom above the largest fit value.

    Returns:
        The top y-limit.
    """
    hi = float(np.nanmax(fit_mean)) if np.isfinite(fit_mean).any() else np.nan
    if not np.isfinite(hi) or hi <= 0:
        return fallback
    return hi * pad


# Panel labels for the TS fit diagnostic, keyed by the fit_output variable
# prefix: (prefix, profile label, gradient label, fallback y-limit).
_TS_FIT_PANELS = (
    ("te", "Te [keV]", "dTe/drho [keV]", 5.0),
    ("ne", "ne [1e20 m^-3]", "dne/drho [1e20 m^-3]", 1.8),
)


def _plot_fit_band(ax, x: np.ndarray, y: np.ndarray, err: np.ndarray, label: str):
    """Plot a fit mean plus its +-1 sigma band over the finite part of the grid.

    Args:
        ax: Axes to plot on.
        x: rho grid.
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
    ax.set_xlabel("rho")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def plot_ts_fits(
    pdf_path: Path | str,
    shot: int,
    ts_time: np.ndarray,
    rho_ch: np.ndarray,
    channel_data: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_output,
    rho_fit: np.ndarray,
    channel_groups: list[tuple[np.ndarray, str, str]] | None = None,
    max_pages: int = 20,
) -> int:
    """Save a PDF comparing the GP fits to the raw TS measurements of one shot.

    One page per sampled measurement time, 2x2 panels: Te (top left) and ne
    (top right) with the channel data, GP fit mean and +-1 sigma predictive
    band, then the GP gradients d/drho with their +-1 sigma bands below.
    Fitted hyperparameters are annotated on the profile panels when the
    method provides them.

    Args:
        pdf_path: Destination PDF path.
        shot: Shot number, for the page titles.
        ts_time: (n_t,) measurement times [s].
        rho_ch: (n_t, n_ch) channel rho locations, NaN where invalid.
        channel_data: {"te": (y, err), "ne": (y, err)}, each (n_t, n_ch), in
            the same units the fit consumed (Te [keV], ne [1e20 m^-3]).
        fit_output: ShotFitOutput whose rows align with ts_time.
        rho_fit: (n_x,) rho grid the fits were predicted on.
        channel_groups: Optional (mask, color, label) triples to split the
            channels by diagnostic; one blue "raw TS" group when None. A mask
            is (n_ch,) for a fixed split (C-Mod core vs edge Thomson) or
            (n_t, n_ch) when the split varies per slice.
        max_pages: Sample the fitted slices down to at most this many pages.

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
    fitted = np.flatnonzero(has_fit)
    # Evenly spread samples rather than a stride, which overshoots max_pages
    # whenever the fitted count is not a multiple of it
    live = fitted[
        np.unique(
            np.linspace(0, fitted.size - 1, min(fitted.size, max_pages)).round()
        ).astype(int)
    ]

    n_ch = rho_ch.shape[1]
    groups = (
        channel_groups
        if channel_groups is not None
        else [(np.ones(n_ch, dtype=bool), "tab:blue", "raw TS")]
    )
    ylims = {
        var: _fit_mean_ylim(getattr(fit_output, f"{var}_fit"), fallback=fallback)
        for var, _, _, fallback in _TS_FIT_PANELS
    }

    with PdfPages(pdf_path) as pdf:
        for i_time in live:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            title = f"shot {shot}  t={ts_time[i_time]:.3f} s"
            for i_var, (var, label, grad_label, _) in enumerate(_TS_FIT_PANELS):
                data_y, err_y = (arr[i_time, :] for arr in channel_data[var])
                rho_at_t = rho_ch[i_time, :]

                ax = axes[0, i_var]
                for mask, color, name in groups:
                    mask_at_t = mask[i_time, :] if mask.ndim == 2 else mask
                    valid = (
                        mask_at_t
                        & np.isfinite(rho_at_t)
                        & np.isfinite(data_y)
                        & np.isfinite(err_y)
                    )
                    if valid.any():
                        ax.errorbar(
                            rho_at_t[valid],
                            data_y[valid],
                            yerr=err_y[valid],
                            fmt="o",
                            ms=4,
                            color=color,
                            label=name,
                            zorder=3,
                        )
                _plot_fit_band(
                    ax,
                    rho_fit,
                    getattr(fit_output, f"{var}_fit")[i_time, :],
                    getattr(fit_output, f"{var}_std")[i_time, :],
                    "GP fit",
                )
                ax.set_ylim(bottom=0, top=ylims[var])
                _style_ts_panel(ax, label, title)
                # The annotation layout is mkgp-specific (5 hyperparameters);
                # other methods' diagnostics are skipped here.
                hyps_all = getattr(fit_output, f"{var}_hyps")
                if (
                    hyps_all is not None
                    and hyps_all.shape[1] == 5
                    and np.isfinite(hyps_all[i_time]).all()
                ):
                    var_h, l1, l2, lw, x0 = hyps_all[i_time]
                    ax.text(
                        0.98,
                        0.98,
                        f"var={var_h:.2f}  l1={l1:.2f}  l2={l2:.2f}\n"
                        f"lw={lw:.2f}  x0={x0:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7,
                        family="monospace",
                    )

                ax = axes[1, i_var]
                _plot_fit_band(
                    ax,
                    rho_fit,
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
