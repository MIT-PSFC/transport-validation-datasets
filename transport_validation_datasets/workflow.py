from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_validation_datasets.gp_fitting.dispatcher import ClusterFitConfig
from transport_validation_datasets.machine.plots import plot_unprocessed_data

# Width of the centered boxcar applied before the transient thresholds are checked [s].
TRANSIENT_SMOOTHING_WINDOW = 5e-3


class DataWorkflow(ABC):
    """Class that handles organization of data processing steps.

    For this study, the general workflow is:
    1. Pull unprocessed data from source and filter down to regions of validity
    - One file per shot
    - Standardized signal names
    - On a common timebase (1 kHz)
    - Slower signals are forward-filled, but tagged as being 'fresh' or not if this is relevant
    - drop regions of invalid data or fix where possible (if plasma current too low, clip negative power to 0, etc.)
    - logging of issues encountered, with plots where relevant to see why
    2. Process and filter data as needed to remove bad shots / fix signals where possible

    3. Combine all shots together into a single xarray Dataset and save to disk
    """

    @property
    @abstractmethod
    def valid_filter(self) -> dict[str, dict[str, float]]:
        """Dictionary of valid ranges for signals, used to filter out invalid data.

        Returns:
            Valid ranges for signals, e.g. {"signal_name": {"min": 0.0, "max": 1.0}}.
        """

    @property
    @abstractmethod
    def transient_filter(self) -> dict[str, float]:
        """Dictionary of thresholds for signals, used to filter out transient events.

        Thresholds are compared against the signal smoothed by a centered boxcar
        TRANSIENT_SMOOTHING_WINDOW wide, NOT the raw signal.

        Returns:
            Thresholds for signals, e.g. {"signal_name": 1.0}.
        """

    @property
    @abstractmethod
    def end_margin(self) -> float:
        """Margin at the end of the shot to ignore when filtering for transients.

        Returns:
            Margin in seconds.
        """

    @property
    @abstractmethod
    def min_pulse_length(self) -> float:
        """Minimum time between the first and last valid ip after filtering.

        Returns:
            Minimum pulse length in seconds.
        """

    @property
    @abstractmethod
    def min_usable_time(self) -> float:
        """Minimum summed duration of the valid ip segments after filtering.

        Returns:
            Minimum usable time in seconds.
        """

    @property
    @abstractmethod
    def shot_blacklist(self) -> list[int]:
        """List of shots to exclude from processing due to known issues.

        Returns:
            List of shot numbers to exclude.
        """

    def __init__(
        self,
        ds_name: str,
        data_assembly_dir: Path,
        shotlist_file: Path | None = None,
        max_num_shots: int | None = None,
        cluster_config: ClusterFitConfig | None = None,
        fit_workers: int | None = 1,
        prepare_workers: int | None = 1,
    ):
        """Set up the dataset directories and resolve the shotlist.

        Args:
            ds_name: Name of the dataset.
            data_assembly_dir: Directory where data files are stored and final
                dataset will be saved.
            shotlist_file: Path to file containing list of shots to process. If None,
                will call _get_shotlist_from_source() to retrieve shotlist from
                device-specific source.
            max_num_shots: Maximum number of shots to process (for testing). If None,
                process all shots.
            cluster_config: If provided, GP profile fitting is dispatched to a SLURM
                cluster (see datasets/gp_fitting/dispatcher.py). If None, fitting
                runs in-process.
            fit_workers: Number of local processes for in-process GP fitting
                (serial mode only).
            prepare_workers: Threads used to stage source data (see stage_shots). Only
                raise it for sources that tolerate concurrent reads: MAST reads public
                S3 and does, disruption_py's MDSplus connections do not.
        """
        self.ds_name = ds_name
        self.data_assembly_dir = data_assembly_dir

        self.max_num_shots = max_num_shots
        if max_num_shots is None:
            self.final_ds_dir = self.data_assembly_dir / ds_name / "dataset_full"
        else:
            self.final_ds_dir = (
                self.data_assembly_dir / ds_name / f"dataset_{max_num_shots}"
            )

        self.cluster_config = cluster_config
        self.fit_workers = fit_workers
        self.prepare_workers = prepare_workers

        # Set up subdirectories for unprocessed data, fit staging, and final dataset
        self.unprocessed_data_dir = data_assembly_dir / "01_unprocessed"
        self.rejected_shots_dir = self.unprocessed_data_dir / "rejected_shots"
        self.accepted_shots_dir = self.unprocessed_data_dir / "accepted_shots"
        self.fit_staging_dir = data_assembly_dir / "02_fit_staging"
        self.fit_results_dir = data_assembly_dir / "03_fit_results"
        self.fit_plots_dir = self.fit_results_dir / "ts_fits"

        if shotlist_file is None:
            logger.info(
                "No shotlist file provided, retrieving shotlist from device-specific source"
            )
            self.shotlist = self._get_shotlist_from_source()
            logger.info(f"Retrieved {len(self.shotlist)} shots from source")
        else:
            with open(shotlist_file) as f:
                lines = f.readlines()
                self.shotlist = [
                    int(line.strip()) for line in lines if line.strip().isdigit()
                ]
            logger.info(f"Loaded {len(self.shotlist)} shots from {shotlist_file}")

    @abstractmethod
    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from device-specific source.

        This method is called when no shotlist file is provided. Subclasses should
        implement their own logic (SQL database query, reading from existing dataset,
        etc.)

        Returns:
            Shot numbers to process.
        """

    @abstractmethod
    def make_unprocessed_data_files(self):
        """Pull data from source and filter down to regions of validity.

        Saves one file per shot in the unprocessed_data_dir. Resulting files have
        standardized signal names and are on a common timebase (1 kHz).

        Also makes plots for each shot to visualize the data and any issues
        encountered.
        """

    def filter_and_plot(self, ds_input: xr.Dataset) -> xr.Dataset | None:
        """Take datasets with standardized names, run filtering on them, and plot results.

        Rejected shots are plotted unfiltered to rejected_shots_dir. Accepted shots
        are plotted unfiltered to accepted_shots_dir, with the kept segments shaded
        green. Both plots mark the end margin cutoff and (if one was found) the
        transient cutoff.

        Args:
            ds_input: Dataset with standardized signal names for one shot.

        Returns:
            The filtered dataset, or None if the shot should be rejected.
        """
        shot = ds_input.shot.values[0]

        # 0: Cut all data end_margin seconds before ip is NaN to avoid including obviously disruptive data at the end of the shot
        ip_valid = ds_input["ip"].notnull().any(dim="shot")
        last_valid_time = float(ip_valid[::-1].idxmax(dim="time"))
        end_margin_time = last_valid_time - self.end_margin
        valid_mask = ds_input["time"] < end_margin_time

        # 1: Apply valid_filter
        for signal, bounds in self.valid_filter.items():
            if "min" in bounds:
                valid_mask = valid_mask & (ds_input[signal] >= bounds["min"])
            if "max" in bounds:
                valid_mask = valid_mask & (ds_input[signal] <= bounds["max"])
            if "min_abs" in bounds:
                valid_mask = valid_mask & (
                    np.abs(ds_input[signal]) >= bounds["min_abs"]
                )
            if "max_abs" in bounds:
                valid_mask = valid_mask & (
                    np.abs(ds_input[signal]) <= bounds["max_abs"]
                )

        # 2: Apply transient_filter: cut everything from the first time a signal exceeds
        # its threshold. The comparison uses each signal smoothed by a centered boxcar
        # (TRANSIENT_SMOOTHING_WINDOW wide) so that sporadic noise spikes on their own do
        # not trip the filter.
        dt = float(np.median(np.diff(ds_input["time"].values)))
        smoothing_samples = max(1, round(TRANSIENT_SMOOTHING_WINDOW / dt))
        if smoothing_samples % 2 == 0:
            # Boxcar must be odd so it stays centered on the present timestep
            smoothing_samples += 1

        transient_margin_time = None
        for signal, threshold in self.transient_filter.items():
            smoothed = (
                ds_input[signal]
                .rolling(time=smoothing_samples, center=True, min_periods=1)
                .mean()
            )
            exceeded = valid_mask & (smoothed > threshold)
            if exceeded.any():
                first_time = float(
                    ds_input["time"].where(exceeded.any(dim="shot")).min()
                )
                if transient_margin_time is None or first_time < transient_margin_time:
                    transient_margin_time = first_time
        if transient_margin_time is not None:
            valid_mask = valid_mask & (ds_input["time"] < transient_margin_time)

        ds_filtered = ds_input.where(valid_mask, drop=True)

        # If there is not sufficient data after filtering, return None to indicate that this shot should be rejected:
        # the first and last non-nan ip must be at least min_pulse_length apart, and the
        # non-nan segments within must sum to at least min_usable_time
        ip_valid_filtered = ds_filtered["ip"].notnull().any(dim="shot")
        ip_times = ds_filtered["time"].values[ip_valid_filtered.values]
        if ip_times.size < 2:
            pulse_length = 0.0
            usable_time = 0.0
        else:
            pulse_length = float(ip_times[-1] - ip_times[0])
            # Timebase is uniform 1 kHz, so any gap beyond 1.5 ms separates two segments
            gaps = np.diff(ip_times)
            usable_time = float(gaps[gaps < 1.5e-3].sum())

        if pulse_length < self.min_pulse_length or usable_time < self.min_usable_time:
            logger.warning(
                f"Shot {shot} rejected: pulse length {pulse_length:.3f} s "
                f"(min {self.min_pulse_length}) or usable time {usable_time:.3f} s "
                f"(min {self.min_usable_time}) insufficient after filtering"
            )
            plot_unprocessed_data(
                ds_input,
                self.rejected_shots_dir / f"shot_{shot}.png",
                title=f"Shot {shot} (REJECTED)",
                valid_filter=self.valid_filter,
                transient_filter=self.transient_filter,
                end_margin_time=end_margin_time,
                transient_margin_time=transient_margin_time,
            )
            return None
        else:
            # Plot the entire shot, with the kept segments shaded green
            kept_times = ds_filtered["time"].values
            breaks = np.flatnonzero(np.diff(kept_times) > 1.5e-3)
            span_starts = np.insert(kept_times[breaks + 1], 0, kept_times[0])
            span_ends = np.append(kept_times[breaks], kept_times[-1])
            plot_unprocessed_data(
                ds_input,
                self.accepted_shots_dir / f"shot_{shot}.png",
                title=f"Shot {shot}",
                valid_filter=self.valid_filter,
                transient_filter=self.transient_filter,
                end_margin_time=end_margin_time,
                transient_margin_time=transient_margin_time,
                kept_spans=list(zip(span_starts.tolist(), span_ends.tolist())),
            )
            return ds_filtered
