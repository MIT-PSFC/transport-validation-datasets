"""Build the packaged MAST shotlist from the FAIR-MAST catalog and the public stores.

A shot of one of CAMPAIGNS makes the list when
1: its stores carry everything get_source_dataset reads (open_shot_sources),
2: its plasma current passes the workflow's ip gates (see _ip_window), and
3: the Thomson chord passes within NEAR_AXIS_RHO_TOR_NORM of the magnetic axis
   for more than NEAR_AXIS_MIN_FRACTION of the time those gates keep.

Relies on the AYC Thomson system, only valid past M7 shot ~23000.

Every scanned shot gets a row in a CSV with the reason it was rejected.
The scan resumes from that CSV, and a shot whose stores could not be reached gets no row,
so running the same command again retries it.

    uv run python -m transport_validation_datasets.machine.mast.shotlist build scratch/mast_shotlist/mast_scan.csv
"""

import csv
import datetime
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np
import xarray as xr
from loguru import logger

from transport_validation_datasets.machine.generic import (
    EQ_MATCH_MAX_PERIODS,
    end_of_shot_index,
    rho_tor_norm_from_psi_n,
    signal_on_grid,
)
from transport_validation_datasets.machine.mast.mast_dataset import (
    DEFAULT_SHOTLIST_FILE,
    TS_CHANNEL_Z,
    MASTDataWorkflow,
    MASTSettings,
    MissingSourceError,
    efm_flux_map,
    open_shot_sources,
)
from transport_validation_datasets.workflow import (
    keep_longest_segment,
    kept_span,
)

# FAIR-MAST shot catalog, pages of at most 100 shots
CATALOG_URL = "https://mastapp.site/json/shots"
CAMPAIGNS = ("M7", "M8", "M9")

# The purpose of this repo is to validate transport codes,
# so we need shots with a Thomson chord that passes near the axis for most of it.
NEAR_AXIS_RHO_TOR_NORM = 0.1
NEAR_AXIS_MIN_FRACTION = 0.8


def _campaign_shots(campaign: str) -> list[int]:
    """Page through the FAIR-MAST catalog for the shots of one campaign.

    Args:
        campaign: Campaign name, e.g. "M9".

    Returns:
        The campaign's shot numbers, in catalog order.
    """
    shots = []
    cursor = None
    while True:
        query = {"filters": f"campaign$eq:{campaign}", "size": 100}
        if cursor is not None:
            query["cursor"] = cursor
        url = f"{CATALOG_URL}?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        shots += [item["shot_id"] for item in page["items"]]
        if page["next_page"] is None:
            return shots
        # next_page comes URL encoded, urlencode would encode it twice
        cursor = urllib.parse.unquote(page["next_page"])


def _ip_window(summary: xr.Dataset, timebase: np.ndarray) -> np.ndarray:
    """Mask the timebase where the workflow's ip gates keep the shot.

    The ip part of DataWorkflow.filter_and_plot:
    before the end-of-shot cut (end_of_shot_index) and above the min_filter ip threshold,
    with only the longest segment kept.
    The other signals can only cut this down,
    so a shot too short here is too short for the workflow.

    Args:
        summary: The level 2 summary group.
        timebase: The shot's uniform 1 kHz timebase [s].

    Returns:
        Mask over timebase, True where the ip gates keep the sample.
    """
    ip_min = MASTDataWorkflow.min_filter["ip"]
    summary_time = summary["time"].values
    ip_source = np.asarray(summary["ip"].values, dtype=float)
    ip_signed = signal_on_grid(summary_time, ip_source, timebase)
    ip = np.abs(ip_signed)
    end_cut_index = end_of_shot_index(ip, timebase, ip_min, MASTDataWorkflow.end_margin)
    if end_cut_index is None:
        return np.zeros(timebase.size, dtype=bool)
    mask_before_end = np.arange(timebase.size) < end_cut_index
    with np.errstate(invalid="ignore"):
        mask_window = mask_before_end & (ip >= ip_min)
    mask_kept, _ = keep_longest_segment(mask_window, timebase)
    return mask_kept


def _chord_rho_tor_norm_min(
    efm: xr.Dataset, ds_thomson: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest rho_tor_norm on the Thomson chord, per reconstruction.

    The chord runs along z = TS_CHANNEL_Z from the innermost to the outermost channel.
    Its smallest psi_N is where it runs tangent to the flux surfaces,
    and the reconstruction's q profile maps that onto rho_tor_norm
    under the same usability rules as map_ts_channels_to_rho_tor_norm.

    Args:
        efm: The level 1 efm group.
        ds_thomson: The shot's usable Thomson slices, for the channel radii.

    Returns:
        (eq_time, rho_min): the times of the reconstructions with a flux map [s],
        and the smallest rho_tor_norm on the chord at each, NaN where it is unusable.
    """
    eq_time = np.asarray(efm["time"].values, dtype=float)
    psi_axis = np.asarray(efm["psi_axis"].values, dtype=float)
    psi_range = np.asarray(efm["psi_boundary"].values, dtype=float) - psi_axis
    psi_map = efm_flux_map(efm)
    psi_chord = psi_map.interp(z=TS_CHANNEL_Z).transpose("time", "major_radius").values
    r_grid = psi_map["major_radius"].values
    ts_r = ds_thomson["ts_channel_r"].values
    mask_on_chord = (r_grid >= np.nanmin(ts_r)) & (r_grid <= np.nanmax(ts_r))
    psi_n_chord = (psi_chord[:, mask_on_chord] - psi_axis[:, None]) / psi_range[:, None]
    qpsi = np.asarray(efm["qpsi_c"].transpose("time", "psi_norm").values, dtype=float)
    sol_extension = MASTSettings().sol_extension

    rho_min = np.full(eq_time.size, np.nan)
    for i in range(eq_time.size):
        if (
            not mask_on_chord.any()
            or not np.isfinite(psi_range[i])
            or np.abs(psi_range[i]) < 1e-10
            or not np.all(np.isfinite(psi_n_chord[i]))
            or not np.all(np.isfinite(qpsi[i]))
        ):
            continue
        psi_n_min = psi_n_chord[i].min()
        rho_min[i] = rho_tor_norm_from_psi_n(psi_n_min, qpsi[i], sol_extension)

    mask_has_flux = np.isfinite(psi_axis)
    return eq_time[mask_has_flux], rho_min[mask_has_flux]


def _near_axis_fraction(
    window_times: np.ndarray, eq_time: np.ndarray, rho_min: np.ndarray
) -> float:
    """Fraction of the window in which the chord passes within NEAR_AXIS_RHO_TOR_NORM of the axis.

    Each window time takes the reconstruction nearest to it,
    accepted within EQ_MATCH_MAX_PERIODS of the reconstruction period
    as when the Thomson channels are mapped.
    A time with no usable reconstruction in reach counts as far from the axis.

    Args:
        window_times: The times the ip gates keep [s].
        eq_time: Reconstruction times [s].
        rho_min: Smallest rho_tor_norm on the chord at each reconstruction.

    Returns:
        The near-axis fraction of window_times.
    """
    if eq_time.size < 2:
        return 0.0
    eq_period = float(np.median(np.diff(eq_time)))
    time_to_each_eq = np.abs(window_times[:, None] - eq_time[None, :])
    nearest = time_to_each_eq.argmin(axis=1)
    time_to_eq = np.abs(eq_time[nearest] - window_times)
    mask_matched = time_to_eq <= EQ_MATCH_MAX_PERIODS * eq_period
    with np.errstate(invalid="ignore"):
        mask_near = mask_matched & (rho_min[nearest] <= NEAR_AXIS_RHO_TOR_NORM)
    return float(mask_near.mean())


def _scan_shot(shot: int) -> dict | None:
    """Decide whether one shot makes the shotlist.

    Args:
        shot: Shot number to scan.

    Returns:
        The shot's scan row, or None when its stores could not be read, worth retrying.
    """
    row = {"shot": shot, "accepted": False, "reason": ""}
    try:
        sources = open_shot_sources(shot)
    except MissingSourceError as e:
        return {**row, "reason": str(e)}
    except Exception as e:
        logger.warning(f"Shot {shot}: failed to read the stores: {e}")
        return None
    if sources is None:
        return None

    mask_window = _ip_window(sources.summary, sources.timebase)
    window_times = sources.timebase[mask_window]
    pulse_length = kept_span(mask_window, sources.timebase)
    row.update(pulse_length=round(pulse_length, 3))
    if pulse_length < MASTDataWorkflow.min_pulse_length:
        return {**row, "reason": "Plasma current window too short."}

    try:
        eq_time, rho_min = _chord_rho_tor_norm_min(sources.efm, sources.ds_thomson)
    except Exception as e:
        logger.warning(f"Shot {shot}: failed to read the equilibrium: {e}")
        return None
    if not np.isfinite(rho_min).any():
        # Also where qpsi_c is NaN
        return {**row, "reason": "No usable reconstruction on the chord."}
    near_axis_fraction = _near_axis_fraction(
        window_times.astype(float), eq_time, rho_min
    )
    rho_min_median = float(np.nanmedian(rho_min))
    row.update(
        near_axis_fraction=round(near_axis_fraction, 3),
        rho_tor_norm_min_median=round(rho_min_median, 3),
    )
    if near_axis_fraction <= NEAR_AXIS_MIN_FRACTION:
        return {**row, "reason": "Thomson chord misses the axis."}
    return {**row, "accepted": True}


class MASTShotlistCLI:
    """Build the packaged MAST shotlist, see the module docstring for the criteria."""

    def build(
        self,
        scan_csv: Path | str,
        shotlist_file: Path | str = DEFAULT_SHOTLIST_FILE,
        workers: int = 16,
    ):
        """Scan every shot of CAMPAIGNS and write the accepted ones to the shotlist.

        Args:
            scan_csv: CSV of per-shot scan rows, appended to and resumed from.
            shotlist_file: Shotlist to write, one shot per line under a comment header.
            workers: Threads scanning shots, the reads are round trip bound.
        """
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        scan_csv = Path(scan_csv)
        shotlist_file = Path(shotlist_file)

        campaign_of_shot = {}
        for campaign in CAMPAIGNS:
            shots = _campaign_shots(campaign)
            logger.info(f"{campaign}: {len(shots)} shots in the catalog")
            campaign_of_shot.update(dict.fromkeys(shots, campaign))

        scanned = set()
        if scan_csv.exists():
            with open(scan_csv, newline="") as f:
                scanned = {int(row["shot"]) for row in csv.DictReader(f)}
        shots_to_scan = sorted(set(campaign_of_shot) - scanned)
        logger.info(f"{len(scanned)} shots already scanned, {len(shots_to_scan)} to go")

        fieldnames = [
            "shot",
            "campaign",
            "accepted",
            "reason",
            "pulse_length",
            "near_axis_fraction",
            "rho_tor_norm_min_median",
        ]
        n_unreachable = 0
        scan_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not scan_csv.exists()
        with (
            open(scan_csv, "a", newline="") as f,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            futures = {pool.submit(_scan_shot, shot): shot for shot in shots_to_scan}
            for n_done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                if row is None:
                    n_unreachable += 1
                else:
                    shot = futures[future]
                    writer.writerow({**row, "campaign": campaign_of_shot[shot]})
                    f.flush()
                if n_done % 100 == 0:
                    logger.info(f"Scanned {n_done} of {len(shots_to_scan)} shots")

        with open(scan_csv, newline="") as f:
            rows = [
                row for row in csv.DictReader(f) if int(row["shot"]) in campaign_of_shot
            ]
        accepted = sorted(int(row["shot"]) for row in rows if row["accepted"] == "True")
        header = (
            f"# MAST {CAMPAIGNS[0]}-{CAMPAIGNS[-1]} shots whose Thomson chord passes within "
            f"rho_tor_norm {NEAR_AXIS_RHO_TOR_NORM} of the magnetic axis "
            f"for over {NEAR_AXIS_MIN_FRACTION:.0%} of the plasma current window.\n"
            f"# Built by machine/mast/shotlist.py on {datetime.date.today().isoformat()}.\n"
        )
        shotlist_file.write_text(header + "".join(f"{shot}\n" for shot in accepted))
        logger.info(
            f"Wrote {len(accepted)} of {len(rows)} scanned shots to {shotlist_file}"
        )
        if n_unreachable:
            logger.warning(
                f"{n_unreachable} shots could not be read, run again to retry them"
            )


if __name__ == "__main__":
    fire.Fire(MASTShotlistCLI)
