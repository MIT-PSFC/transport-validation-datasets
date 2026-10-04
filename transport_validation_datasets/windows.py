"""Time windows from a shotlist, and how they restrict or pool the fit inputs.

A shotlist may carry per-shot time windows (columns shot, t_start, t_end,
a shot on as many rows as it has windows). Windows enter the workflow at
three points: DataWorkflow.__init__ parses them (read_shotlist),
stage_fit_batches restricts or pools a shot's per-sample fit input to them
(restrict_to_windows, pool_windows), and the stack stage places the fitted
profiles on the grid by window (window_membership, in_any_window). The fit
workers never see them: a pooled window is one input row with more columns.
window_bounds and window_centers are the one array form and the one center
rule every caller shares.
"""

import csv
from pathlib import Path

import numpy as np
from loguru import logger

from transport_validation_datasets.gp_fitting.batch_io import ShotFitInput

# Pooled points per window above which the fit is flagged: every GP likelihood
# evaluation costs n^3, times the optimizer restarts and the LOO cleaning passes.
POOLED_POINTS_WARN = 1000

# Widening of the inclusive window bounds [s]. Grid and Thomson sample times are float32
# and the shotlist bounds float64 (float32(0.65) < 0.65), so a sample sitting
# exactly on a bound must not fall out of its window.
WINDOW_TIME_TOL = 1e-6

# Two windows of one shot may overlap, but their centers must sit at least
# this far apart [s], one grid step: the window-averaged profile is labelled at
# the grid time nearest the center, and two windows must not label the same
# grid time.
MIN_CENTER_SEPARATION = 1e-3

WINDOW_COLUMNS = ("t_start", "t_end")
# The shot column, by preference: shot, else pulse_no (what scenario tables
# exported from other tools tend to call it)
SHOT_COLUMNS = ("shot", "pulse_no")


def read_shotlist(
    path: Path | str,
) -> tuple[list[int], dict[int, list[tuple[float, float]]] | None]:
    """Read a shotlist file, with or without time windows.

    Two formats.
    Plain: one shot number per line, blank lines skipped, or a CSV with a
    shot column and no window columns, other columns ignored.
    Windowed: a CSV whose header holds shot, t_start and t_end [s], other
    columns ignored; a shot appears on one row per window.
    The shot column may also be called pulse_no.
    The header is read as plain comma-separated names, so a quoted header,
    a byte order mark or another separator leaves the file plain,
    where its lines are not shot numbers and the error names the first one.

    Args:
        path: The shotlist file.

    Returns:
        (shotlist, windows): the unique shots in order of first appearance,
        and per shot its windows sorted by start time, or None for a plain
        shotlist.

    Windows of one shot may overlap. Two may not share a center (closer than
    MIN_CENTER_SEPARATION), that is where the window-averaged profile is
    labelled.

    Raises:
        ValueError: If a line is not a shot number (plain) or a row's shot
            column is not an integer (CSV), a windowed file has no shot
            column, a window is not a finite range with t_start < t_end,
            two windows of one shot have the same center, or the file
            holds no shot at all.
    """
    path = Path(path)
    with open(path, newline="") as f:
        header = [column.strip() for column in f.readline().split(",")]
        f.seek(0)
        shot_column = next((c for c in SHOT_COLUMNS if c in header), None)
        if not all(column in header for column in WINDOW_COLUMNS):
            if shot_column is None:
                plain = _read_plain_shot_lines(f, path)
            else:
                plain = []
                for line_number, row in enumerate(csv.DictReader(f), start=2):
                    try:
                        plain.append(int(row[shot_column]))
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"{path} line {line_number}: {e}") from e
            if not plain:
                raise ValueError(f"{path}: no shot in the shotlist")
            return list(dict.fromkeys(plain)), None

        if shot_column is None:
            raise ValueError(
                f"{path}: a windowed shotlist needs a {' or '.join(SHOT_COLUMNS)} "
                f"column, the header is {header}"
            )
        shots: list[int] = []
        windows: dict[int, list[tuple[float, float]]] = {}
        for line_number, row in enumerate(csv.DictReader(f), start=2):
            try:
                shot = int(row[shot_column])
                t_start = float(row["t_start"])
                t_end = float(row["t_end"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path} line {line_number}: {e}") from e
            if not (np.isfinite(t_start) and np.isfinite(t_end)) or t_start >= t_end:
                raise ValueError(
                    f"{path} line {line_number}: shot {shot} window "
                    f"[{t_start}, {t_end}] is not a finite range with t_start < t_end"
                )
            shots.append(shot)
            windows.setdefault(shot, []).append((t_start, t_end))
    if not shots:
        raise ValueError(f"{path}: no shot in the shotlist")

    for shot, shot_windows in windows.items():
        shot_windows.sort()
        centers = np.sort(window_centers(shot_windows))
        for earlier, later in zip(centers, centers[1:]):
            if later - earlier < MIN_CENTER_SEPARATION:
                raise ValueError(
                    f"{path}: shot {shot} has two windows centered at {earlier} s "
                    f"and {later} s, closer than {MIN_CENTER_SEPARATION} s: they "
                    f"would label the same grid time"
                )
    return list(dict.fromkeys(shots)), windows


def _read_plain_shot_lines(lines, path: Path) -> list[int]:
    """Read one shot number per line, skipping blank lines.

    Args:
        lines: The open file.
        path: Its path, for the message.

    Returns:
        The shot numbers in file order, repeats included.

    Raises:
        ValueError: If a non-blank line is not an integer.
    """
    shots = []
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            shots.append(int(text))
        except ValueError as e:
            raise ValueError(
                f"{path} line {line_number}: {text!r} is not a shot number. "
                "A shotlist is one shot number per line, or a CSV with a shot "
                "(or pulse_no) column and optional t_start and t_end columns."
            ) from e
    return shots


def window_bounds(windows) -> np.ndarray:
    """Put a window list in its array form.

    Args:
        windows: Window bounds [s] as (start, end) pairs, a list of tuples or
            an array, empty for none.

    Returns:
        (n_w, 2) float64 array.
    """
    return np.asarray(windows, dtype=float).reshape(-1, 2)


def window_centers(windows) -> np.ndarray:
    """Center of each window, where the window-averaged profile is labelled.

    The one center rule: the pooled row's time, the fresh_profile placement
    in the store and the shotlist's same-center check all take it from here.

    Args:
        windows: Window bounds [s], any form window_bounds accepts.

    Returns:
        (n_w,) centers [s].
    """
    return window_bounds(windows).mean(axis=1)


def window_membership(times: np.ndarray, windows) -> np.ndarray:
    """Find which windows hold each time.

    Bounds are inclusive, widened by WINDOW_TIME_TOL. Windows may overlap, so
    a time can sit in several. This is the one membership rule the fit
    staging and the stack stage share, so a Thomson sample is fit in the
    windows its grid time is stored under.

    Args:
        times: Times to place [s], any shape.
        windows: (n_w, 2) window bounds [s].

    Returns:
        (*times.shape, n_w) mask, True where the window holds the time.
    """
    times = np.asarray(times, dtype=float)
    bounds = window_bounds(windows)
    return (times[..., None] >= bounds[:, 0] - WINDOW_TIME_TOL) & (
        times[..., None] <= bounds[:, 1] + WINDOW_TIME_TOL
    )


def in_any_window(times: np.ndarray, windows) -> np.ndarray:
    """Find which times fall inside at least one window.

    Args:
        times: Times to place [s], any shape.
        windows: Window bounds [s], any form window_bounds accepts.

    Returns:
        Mask of times.shape, True where some window holds the time.
    """
    return window_membership(times, windows).any(axis=-1)


def restrict_to_windows(fit_input: ShotFitInput, windows) -> ShotFitInput | None:
    """Keep only the Thomson sample rows of a fit input that fall inside a window.

    A sample inside several overlapping windows is kept once.

    Args:
        fit_input: One shot's per-sample fit input.
        windows: The shot's (n_w, 2) window bounds [s], sorted by start.

    Returns:
        The rows inside the windows, tagged with the shot's window list and,
        per row, the earliest window holding it, or None when no Thomson
        sample falls in any window.
    """
    bounds = window_bounds(windows)
    member = window_membership(fit_input.time, bounds)
    keep = np.flatnonzero(member.any(axis=1))
    if keep.size == 0:
        return None
    return ShotFitInput(
        x=fit_input.x[keep],
        te_y=fit_input.te_y[keep],
        te_err=fit_input.te_err[keep],
        ne_y=fit_input.ne_y[keep],
        ne_err=fit_input.ne_err[keep],
        time=fit_input.time[keep],
        windows=bounds,
        window_index=member[keep].argmax(axis=1),
    )


def pool_windows(fit_input: ShotFitInput, windows, shot: int) -> ShotFitInput:
    """Pool the Thomson samples of each window into one row, to be fit as one profile.

    The row of a window is its sample rows laid end to end, so a channel
    appears once per sample, at that sample's rho_tor_norm. Rows are NaN padded to the
    widest window in whole samples, so a per-channel mask still tiles onto
    them. The row's time is the window center. A window with no sample keeps
    an all-NaN row, which the worker skips, so the rows stay aligned with the
    window list. A sample inside overlapping windows is pooled into each of
    them. Nothing caps the pooled size: a window with more than
    POOLED_POINTS_WARN points is logged as critical, the GP cost grows as n^3.

    Args:
        fit_input: One shot's per-sample fit input.
        windows: The shot's (n_w, 2) window bounds [s], sorted by start.
        shot: Shot number, for the log lines.

    Returns:
        The pooled fit input, one row per window.
    """
    bounds = window_bounds(windows)
    member = window_membership(fit_input.time, bounds)
    n_ch = fit_input.x.shape[1]
    rows = [np.flatnonzero(member[:, w]) for w in range(bounds.shape[0])]
    width = max(1, max(samples.size for samples in rows)) * n_ch

    def pooled(values: np.ndarray) -> np.ndarray:
        out = np.full((bounds.shape[0], width), np.nan, dtype=values.dtype)
        for w, samples in enumerate(rows):
            if samples.size:
                out[w, : samples.size * n_ch] = values[samples].reshape(-1)
        return out

    x = pooled(fit_input.x)
    for w, samples in enumerate(rows):
        start, end = bounds[w]
        if samples.size == 0:
            logger.warning(
                f"Shot {shot}: no Thomson sample in window [{start:.3f}, {end:.3f}] s, "
                "its row stays empty"
            )
            continue
        n_points = int(np.isfinite(x[w]).sum())
        message = (
            f"Shot {shot}: window [{start:.3f}, {end:.3f}] s pools {samples.size} "
            f"Thomson samples, {n_points} points with a finite rho_tor_norm"
        )
        if n_points > POOLED_POINTS_WARN:
            logger.critical(
                f"{message}. Above {POOLED_POINTS_WARN}: every GP likelihood "
                "evaluation costs n^3, times restarts and cleaning passes, so "
                "this fit may take very long or run out of memory. Shorten the "
                "window or fit it per sample."
            )
        else:
            logger.info(message)

    return ShotFitInput(
        x=x,
        te_y=pooled(fit_input.te_y),
        te_err=pooled(fit_input.te_err),
        ne_y=pooled(fit_input.ne_y),
        ne_err=pooled(fit_input.ne_err),
        time=window_centers(bounds),
        windows=bounds,
        window_index=np.arange(bounds.shape[0]),
    )
