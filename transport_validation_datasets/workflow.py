import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_validation_datasets import TIME_COORD, TIME_DIM
from transport_validation_datasets.gp_fitting import registry
from transport_validation_datasets.gp_fitting.batch_io import (
    STATUS_NAMES,
    STATUS_SKIPPED,
    FitBatch,
    ShotFitInput,
    default_fit_bounds,
    pack_fit_batch,
    unpack_fit_batch,
    unpack_fit_results,
)
from transport_validation_datasets.gp_fitting.dispatcher import (
    ClusterFitConfig,
    plan_batches,
)
from transport_validation_datasets.machine.plots import (
    plot_ts_fits,
    plot_unprocessed_data,
)

# Width of the centered boxcar applied before the transient thresholds are checked [s].
TRANSIENT_SMOOTHING_WINDOW = 5e-3

# Shots per staged batch when fitting locally: one, so a slow serial run can
# resume shot by shot. Cluster runs use ClusterFitConfig.shots_per_batch.
LOCAL_SHOTS_PER_BATCH = 1

# The radial coordinate every profile is fit on.
RHO_DEFINITION = (
    "Normalized outboard midplane minor radius: 0 at the magnetic axis, 1 at the LCFS."
    "See machine.generic.map_ts_channels_to_rho."
)


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

    # GP fit staging knobs
    fit_rho = np.linspace(0.0, 1.0, 51)
    fit_min_points = 10
    fit_scale_per_slice = False
    fit_bounds = default_fit_bounds()

    def __init__(
        self,
        ds_name: str,
        data_assembly_dir: Path,
        shotlist_file: Path | None = None,
        max_num_shots: int | None = None,
        cluster_config: ClusterFitConfig | None = None,
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
                cluster (see gp_fitting/dispatcher.py).
                If None, fitting runs single-threaded in this process.
            prepare_workers: Threads used to stage source data.
                Should only be > 1 for sources that tolerate concurrent reads
                (MAST reads public S3 and does, disruption_py's MDSplus connections do not)
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
        self.prepare_workers = prepare_workers

        # Set up subdirectories for unprocessed data, fit staging, and final dataset
        self.unprocessed_data_dir = data_assembly_dir / "01_unprocessed"
        self.rejected_shots_dir = self.unprocessed_data_dir / "rejected_shots"
        self.accepted_shots_dir = self.unprocessed_data_dir / "accepted_shots"
        self.failed_shots_dir = self.unprocessed_data_dir / "failed_shots"
        self.fit_staging_dir = data_assembly_dir / "02_fit_staging"
        self.fit_batches_dir = self.fit_staging_dir / "batches"
        self.failed_fits_dir = self.fit_staging_dir / "failed_shots"
        self.fit_results_dir = data_assembly_dir / "03_fit_results"
        self.fit_plots_dir = self.fit_results_dir / "ts_fits"

        # Log the run to a timestamped file named for when it was launched.
        self.logs_dir = data_assembly_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        launch_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logs_dir / f"{ds_name}_{launch_time}.log"
        logger.add(self.log_file)
        logger.info(f"Logging this run to {self.log_file}")

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

    def record_failed_shot(self, shot: int, reason: str):
        """Record that a shot failed to produce an unprocessed file.

        Writes failed_shots_dir/<shot>.txt with the reason so later runs skip the
        shot via shot_already_failed instead of retrying it.

        Args:
            shot: Shot number that failed.
            reason: Human-readable reason the shot was skipped.
        """
        self.failed_shots_dir.mkdir(parents=True, exist_ok=True)
        (self.failed_shots_dir / f"{shot}.txt").write_text(reason)

    def shot_already_failed(self, shot: int) -> bool:
        """Check whether a shot was recorded as failed on a previous run.

        Args:
            shot: Shot number to check.

        Returns:
            True if a failure record exists for the shot, False otherwise.
        """
        return (self.failed_shots_dir / f"{shot}.txt").exists()

    def record_failed_fit(self, shot: int, reason: str):
        """Record that a shot could not be staged for GP fitting.

        Writes failed_fits_dir/<shot>.txt with the reason so later runs skip
        the shot via fit_already_failed instead of re-staging it. Only
        permanent per-shot problems belong here (no fittable channel data),
        transient cluster failures are retried on the next run instead.

        Args:
            shot: Shot number that failed.
            reason: Human-readable reason the shot was skipped.
        """
        self.failed_fits_dir.mkdir(parents=True, exist_ok=True)
        (self.failed_fits_dir / f"{shot}.txt").write_text(reason)

    def fit_already_failed(self, shot: int) -> bool:
        """Check whether a shot was recorded as unfittable on a previous run.

        Args:
            shot: Shot number to check.

        Returns:
            True if a fit-failure record exists for the shot, False otherwise.
        """
        return (self.failed_fits_dir / f"{shot}.txt").exists()

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
                self.rejected_shots_dir / f"{shot}.png",
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
                self.accepted_shots_dir / f"{shot}.png",
                title=f"Shot {shot}",
                valid_filter=self.valid_filter,
                transient_filter=self.transient_filter,
                end_margin_time=end_margin_time,
                transient_margin_time=transient_margin_time,
                kept_spans=list(zip(span_starts.tolist(), span_ends.tolist())),
            )
            return ds_filtered

    @abstractmethod
    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        Subclasses map the TS channels onto rho, convert to the fit units
        (Te [keV], ne [1e20 m^-3]), and apply their device-specific channel
        quality screens and error floors.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset (one 01_unprocessed file).

        Returns:
            The fit input, or None when the shot has nothing fittable (the
            caller records it as failed).
        """

    def run_gp_fitting(self, method: str):
        """Run GP profile fitting on the unprocessed data files.

        The stages, each skipping work that already exists on disk:
        1. Stage: map TS channels onto rho, apply device cleaning
           (prepare_fit_input), and pack batch npz files into
           fit_staging_dir/batches. The staged batches are method-agnostic.
        2. Fit each batch with the chosen method's worker: single-threaded in
           this process when cluster_config is None, otherwise dispatched to
           the SLURM cluster.
        3. Collect the batch results into one dataset,
           fit_results_dir/<method>/fit_results.nc.
        4. Plot the fits per shot into fit_plots_dir/<method>/.

        Args:
            method: Fitting method name (see gp_fitting.registry).
        """
        shots = sorted(int(p.stem) for p in self.unprocessed_data_dir.glob("*.nc"))
        skipped = [s for s in shots if self.fit_already_failed(s)]
        if skipped:
            logger.info(
                f"Skipping {len(skipped)} shots that failed fit staging on a previous run"
            )
        shots = [s for s in shots if s not in set(skipped)]
        logger.info(f"GP fitting {len(shots)} shots with method '{method}'")

        batches = self.stage_fit_batches(shots)
        if self.cluster_config is None:
            self._fit_batches_local(method, batches)
        else:
            from transport_validation_datasets.gp_fitting.dispatcher import (
                ClusterFitDispatcher,
            )

            dispatcher = ClusterFitDispatcher(
                self.cluster_config, self.ds_name, self.fit_batches_dir, method
            )
            dispatcher.run(batches)
        self.collect_fit_results(method)
        self.plot_fit_results(method)

    def stage_fit_batches(self, shots: list[int]) -> dict[str, list[int]]:
        """Stage fit inputs for the given shots into batch npz files.

        Shots already covered by an existing batch file keep their batch
        (and with it the batch id a restarted cluster run lines up on),
        the rest are packed into new batches.
        A shot whose prepare_fit_input returns None is recorded as failed and skipped on later runs.

        Args:
            shots: Shot numbers with unprocessed data files.

        Returns:
            Mapping of batch id to the staged shots in that batch.
        """
        shots_per_batch = (
            self.cluster_config.shots_per_batch
            if self.cluster_config is not None
            else LOCAL_SHOTS_PER_BATCH
        )
        planned = plan_batches(
            self.ds_name, shots, self.fit_batches_dir, shots_per_batch
        )

        batches: dict[str, list[int]] = {}
        for batch_id, batch_shots in sorted(planned.items()):
            in_path = self._batch_in_path(batch_id)
            if in_path.exists():
                batches[batch_id] = batch_shots
                continue
            shot_inputs = {}
            for shot in batch_shots:
                ds = xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc")
                fit_input = self.prepare_fit_input(shot, ds)
                if fit_input is None:
                    self.record_failed_fit(shot, "No fittable Thomson channel data.")
                    continue
                shot_inputs[shot] = fit_input
            if not shot_inputs:
                continue
            pack_fit_batch(
                in_path,
                FitBatch(
                    shot_inputs=shot_inputs,
                    x_star=self.fit_rho,
                    min_points=self.fit_min_points,
                    scale_per_slice=self.fit_scale_per_slice,
                    bounds=self.fit_bounds,
                ),
            )
            batches[batch_id] = sorted(shot_inputs)
            logger.info(f"Staged batch {batch_id} with {len(shot_inputs)} shots")
        return batches

    def _batch_in_path(self, batch_id: str) -> Path:
        return self.fit_batches_dir / f"batch_{batch_id}.npz"

    def _batch_out_path(self, batch_id: str, method: str) -> Path:
        return self.fit_batches_dir / f"batch_{batch_id}_out_{method}.npz"

    def _fit_batches_local(self, method: str, batches: dict[str, list[int]]):
        """Fit staged batches serially in this process, one at a time.

        Args:
            method: Fitting method name.
            batches: Mapping of batch id to shots, from stage_fit_batches.
        """
        for i, batch_id in enumerate(sorted(batches)):
            out_path = self._batch_out_path(batch_id, method)
            if out_path.exists():
                continue
            logger.info(
                f"Fitting batch {batch_id} ({batches[batch_id]}) locally "
                f"({i + 1}/{len(batches)})"
            )
            registry.run_batch_file(method, self._batch_in_path(batch_id), out_path)

    def collect_fit_results(self, method: str) -> Path | None:
        """Collect every batch's fit results into one dataset on disk.

        Rebuilt from the batch result files on every call (cheap) and written
        atomically to fit_results_dir/<method>/fit_results.nc
        Profiles are converted back from the fit units to SI (Te [eV], ne [m^-3])
        to match the unprocessed files' conventions, gradients are per unit rho.
        Batches without a result file yet are skipped with a warning, so a
        partially fit dataset still collects.

        Args:
            method: Fitting method name.

        Returns:
            Path of the written dataset, or None when no results exist yet.

        Raises:
            ValueError: If batches were fit on different rho grids, or a
                batch's result rows do not align with its slice times.
        """
        outputs = {}
        x_star = None
        for in_path in sorted(self.fit_batches_dir.glob("batch_*.npz")):
            if "_out_" in in_path.name:
                continue
            batch_id = in_path.stem.removeprefix("batch_")
            out_path = self._batch_out_path(batch_id, method)
            if not out_path.exists():
                logger.warning(
                    f"No {method} results for batch {batch_id} yet, skipping its shots"
                )
                continue
            with np.load(out_path) as data:
                batch_x_star = data["x_star"]
            if x_star is None:
                x_star = batch_x_star
            elif not np.array_equal(x_star, batch_x_star):
                raise ValueError(
                    f"Batch {batch_id} was fit on a different rho grid; re-stage and refit"
                )
            for shot, so in unpack_fit_results(out_path).items():
                if so.te_fit.shape != (so.time.size, x_star.size):
                    raise ValueError(
                        f"Batch {batch_id} shot {shot}: result rows do not align with its slice times"
                    )
                outputs[shot] = so
        if not outputs:
            logger.warning(f"No {method} fit results to collect")
            return None

        shots = sorted(outputs)
        n_t = max(outputs[s].time.size for s in shots)

        def padded(name: str, fill: float, dtype) -> np.ndarray:
            """Stack one per-shot array over shots, padded to n_t slices.

            Args:
                name: ShotFitOutput attribute to stack.
                fill: Pad value.
                dtype: Output dtype.

            Returns:
                The stacked (shot, time_idx, ...) array.
            """
            first = getattr(outputs[shots[0]], name)
            arr = np.full((len(shots), n_t) + first.shape[1:], fill, dtype=dtype)
            for i, s in enumerate(shots):
                a = getattr(outputs[s], name)
                arr[i, : a.shape[0], ...] = a
            return arr

        status_attrs = {
            "description": "Per-slice fit status",
            "codes": ", ".join(f"{k}={v}" for k, v in STATUS_NAMES.items()),
        }
        data_vars = {}
        for var, si_factor, name, unit, desc in (
            ("te", 1.0e3, "t_e", "eV", "electron temperature"),
            ("ne", 1.0e20, "n_e", "m^-3", "electron density"),
        ):
            for suffix, out_suffix, extra in (
                ("fit", "", ""),
                ("std", "_error", "1-sigma predictive uncertainty of the "),
                ("grad", "_gradient", "d/drho gradient of the "),
                (
                    "grad_std",
                    "_gradient_error",
                    "1-sigma uncertainty of the d/drho gradient of the ",
                ),
            ):
                grad_unit = unit if suffix in ("fit", "std") else f"{unit} per unit rho"
                data_vars[f"{name}{out_suffix}"] = (
                    ("shot", TIME_DIM, "rho"),
                    padded(f"{var}_{suffix}", np.nan, np.float32) * si_factor,
                    {
                        "description": f"{extra}GP-fitted {desc} profile",
                        "units": grad_unit,
                    },
                )
            data_vars[f"{name}_fit_status"] = (
                ("shot", TIME_DIM),
                padded(f"{var}_status", STATUS_SKIPPED, np.int8),
                status_attrs,
            )
            if all(getattr(outputs[s], f"{var}_hyps") is not None for s in shots):
                data_vars[f"{name}_hyperparameters"] = (
                    ("shot", TIME_DIM, "hyperparameter"),
                    padded(f"{var}_hyps", np.nan, np.float32),
                    {"description": f"Fitted GP hyperparameters of the {desc} fit"},
                )

        coords = {
            "shot": shots,
            "rho": ("rho", x_star, {"description": RHO_DEFINITION}),
            TIME_COORD: (
                ("shot", TIME_DIM),
                padded("time", np.nan, np.float32),
                {"units": "s", "description": "Thomson slice times"},
            ),
        }
        worker = registry.load_worker(method)
        hyp_names = getattr(worker, "HYP_NAMES", None)
        if hyp_names is not None and any("hyperparameters" in k for k in data_vars):
            coords["hyperparameter"] = list(hyp_names)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "fit_method": method,
                "rho_definition": RHO_DEFINITION,
                "dataset_name": self.ds_name,
            },
        )
        results_path = self.fit_results_dir / method / "fit_results.nc"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = results_path.with_suffix(".nc.tmp")
        ds.to_netcdf(tmp_path)
        os.replace(tmp_path, results_path)
        logger.info(f"Collected {method} fits for {len(shots)} shots to {results_path}")
        return results_path

    def _fit_plot_channel_groups(self, shot: int) -> list | None:
        """Get the channel grouping used to color the fit plots.

        Subclasses can split channels by diagnostic (e.g. C-Mod core vs edge
        Thomson); the base implementation plots them as one group.

        Args:
            shot: Shot number being plotted.

        Returns:
            (mask, color, label) triples, or None for a single group.
        """
        return None

    def plot_fit_results(self, method: str):
        """Plot the GP fits of every fitted shot, one PDF per shot.

        Plots the exact (cleaned, floored) channel data the fit consumed,
        straight from the staged batch files. Skips shots whose PDF already
        exists.

        Args:
            method: Fitting method name.
        """
        plots_dir = self.fit_plots_dir / method
        for in_path in sorted(self.fit_batches_dir.glob("batch_*.npz")):
            if "_out_" in in_path.name:
                continue
            batch_id = in_path.stem.removeprefix("batch_")
            out_path = self._batch_out_path(batch_id, method)
            if not out_path.exists():
                continue
            batch = None
            for shot, so in unpack_fit_results(out_path).items():
                pdf_path = plots_dir / f"{shot}.pdf"
                if pdf_path.exists():
                    continue
                if batch is None:
                    batch = unpack_fit_batch(in_path)
                si = batch.shot_inputs[shot]
                n_pages = plot_ts_fits(
                    pdf_path,
                    shot,
                    ts_time=si.time,
                    rho_ch=si.x,
                    channel_data={
                        "te": (si.te_y, si.te_err),
                        "ne": (si.ne_y, si.ne_err),
                    },
                    fit_output=so,
                    rho_fit=batch.x_star,
                    channel_groups=self._fit_plot_channel_groups(shot),
                )
                logger.info(f"Plotted {n_pages} fit pages for shot {shot}")
