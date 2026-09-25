import json
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from loguru import logger

from transport_validation_datasets import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_validation_datasets.cleaning import (
    clean_fit_rows,
    drop_rows_without_core,
)
from transport_validation_datasets.dataset_utils import build_tensorized_dataset
from transport_validation_datasets.gp_fitting import registry
from transport_validation_datasets.gp_fitting.batch_io import (
    FIT_MODE_SAMPLE,
    FIT_MODE_WINDOW_AVERAGE,
    FIT_MODE_WINDOW_SAMPLE,
    FIT_VARIABLES,
    STATUS_NAMES,
    STATUS_OK,
    STATUS_REPAIRED,
    FitAnchors,
    FitBatch,
    ShotFitInput,
    ShotFitOutput,
    default_fit_bounds,
    pack_fit_batch,
    read_batch_anchors,
    read_batch_setting,
    read_batch_shots,
    read_batch_windows,
    unpack_fit_batch,
    unpack_fit_results,
)
from transport_validation_datasets.gp_fitting.dispatcher import (
    ClusterFitConfig,
    plan_batches,
)
from transport_validation_datasets.machine.generic import (
    SOL_EXTENSIONS,
    efit_cocos_from_signs,
    standardize_signal_attrs,
)
from transport_validation_datasets.machine.plots import (
    plot_ts_fits,
    plot_unprocessed_data,
)
from transport_validation_datasets.provenance import (
    SOURCE_VOLATILE_KEYS,
    build_provenance,
    build_stamp,
    merge_shot_attrs,
    source_provenance,
    to_json,
)
from transport_validation_datasets.windows import (
    in_any_window,
    pool_windows,
    read_shotlist,
    restrict_to_windows,
    window_bounds,
    window_centers,
    window_membership,
)

# Width of the centered boxcar applied before the transient thresholds are checked [s].
TRANSIENT_SMOOTHING_WINDOW = 5e-3

# Shots per staged batch when fitting locally: one, so a slow serial run can
# resume shot by shot. Cluster runs use ClusterFitConfig.shots_per_batch.
LOCAL_SHOTS_PER_BATCH = 1

# The radial coordinate every profile is fit on.
RHO_TOR_NORM_DEFINITION = (
    "Normalized toroidal flux coordinate rho_tor_norm = sqrt(Phi_N): 0 at the magnetic axis, 1 at the LCFS. "
    "Outside the LCFS Phi_N continues linearly in psi_N, see the sol_extension attribute."
)

# The fit grid runs past this so every anchor is fit and plotted.
# The fit files, the stores and the IMAS export keep only the grid up to here.
STORED_RHO_TOR_NORM_MAX = 1.1

# Per-slice fit statuses that count as a usable profile, see gp_fitting.batch_io.
USABLE_FIT_STATUSES = (STATUS_OK, STATUS_REPAIRED)

# A grid time carries a sample of its own when it sits this close to one [s].
# Only absorbs float round-off, everything shares the staged 1 kHz timebase.
SAMPLE_TIME_TOL = 1e-6

# How long a slowly sampled signal (a fitted profile, an equilibrium) is held
# forward onto the 1 kHz timebase, in periods of its own sampling. Above 1 to
# tolerate jitter in the sampling, low enough that nothing is carried across a
# real gap: the end of the shot, or a stretch the filtering cut away.
MAX_HOLD_PERIODS = 1.5

# Unprocessed signals carried into the internal dataset. The union over every
# device: a signal the device does not have comes through as NaN, so all the
# devices' datasets share one schema.
DATASET_0D_SIGNALS = (
    "ip",
    "b0",
    "energy_mhd",
    "beta_tor_norm",
    "n_e_line_average",
    "minor_radius",
    "geometric_axis_r",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "power_ohm",
    "power_radiated",
    "power_nbi",
    "power_ic",
    "power_lh",
)

# GEQDSK block, everything needed to rebuild the equilibrium of a slice.
# See machine.generic.make_geqdsk_dataset. The five that are constant in time
# (rcentr, rleft, rdim, zmid, zdim) and the limiter contour are carried per
# slice like the rest: they compress to nothing and keep the layout uniform.
DATASET_EQUILIBRIUM_SIGNALS = (
    "rmagx",
    "zmagx",
    "simagx",
    "sibdry",
    "bcentr",
    "current",
    "rcentr",
    "rleft",
    "rdim",
    "zmid",
    "zdim",
    "fpol",
    "pres",
    "ffprime",
    "pprime",
    "qpsi",
    "psirz",
    "rbdry",
    "zbdry",
    "rlim",
    "zlim",
)

# The raw Thomson channel measurements and chord geometry.
# In the internal dataset, stripped from the published one by default
RAW_TS_CHANNEL_SIGNALS = (
    "ts_channel_r",
    "ts_channel_z",
    "ts_channel_t_e",
    "ts_channel_t_e_error",
    "ts_channel_n_e",
    "ts_channel_n_e_error",
)


@dataclass(frozen=True)
class DeviceSettings:
    """Device-specific settings of a workflow, the [<device>] table of the config file.

    A device subclasses this to add settings or change defaults,
    and points its workflow's settings_cls at the subclass.
    config.py builds the instance from the TOML table by field name.

    The anchors are virtual observations every fit method adds to every slice,
    in the fit units (Te [keV], ne [1e20 m^-3]), gradients per unit rho_tor_norm.
    The outer anchors sit at 1.3 to 1.6, past the SOL channels,
    which rho_tor_norm stretches out to ~1.25.

    Attributes:
        sol_extension: How the Thomson channel mapping continues Phi_N outside the LCFS
        pedestal_rho_tor_norm: Pedestal location, fixed for every slice of Te and ne.
            zk places its kernel's length-scale transition there, akho its mtanh.
        te_value_anchors: Rows of [rho_tor_norm, Te, error].
        te_grad_anchors: Rows of [rho_tor_norm, dTe/drho_tor_norm, error].
        ne_value_anchors: Rows of [rho_tor_norm, ne, error].
        ne_grad_anchors: Rows of [rho_tor_norm, dne/drho_tor_norm, error].
    """

    sol_extension: str = "secant"
    pedestal_rho_tor_norm: float = 1.0
    te_value_anchors: list = field(
        default_factory=lambda: [
            [1.3, 0.0, 0.01],
            [1.4, 0.0, 0.01],
            [1.5, 0.0, 0.01],
            [1.6, 0.0, 0.01],
        ]
    )
    te_grad_anchors: list = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.1],
            [1.4, 0.0, 0.1],
            [1.5, 0.0, 0.1],
            [1.6, 0.0, 0.1],
        ]
    )
    ne_value_anchors: list = field(
        default_factory=lambda: [
            [1.3, 0.0, 0.01],
            [1.4, 0.0, 0.01],
            [1.5, 0.0, 0.01],
            [1.6, 0.0, 0.01],
        ]
    )
    ne_grad_anchors: list = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.1],
            [1.4, 0.0, 0.1],
            [1.5, 0.0, 0.1],
            [1.6, 0.0, 0.1],
        ]
    )

    def __post_init__(self):
        if self.sol_extension not in SOL_EXTENSIONS:
            raise ValueError(
                f"sol_extension must be one of {SOL_EXTENSIONS}, got {self.sol_extension!r}"
            )
        for name in (
            "te_value_anchors",
            "te_grad_anchors",
            "ne_value_anchors",
            "ne_grad_anchors",
        ):
            raw_rows = getattr(self, name)
            rows = np.asarray(raw_rows, dtype=float)
            if rows.size and (rows.ndim != 2 or rows.shape[1] != 3):
                raise ValueError(
                    f"{name} must be rows of [rho_tor_norm, value, error], got {raw_rows!r}"
                )


class DataWorkflow(ABC):
    """Device-independent stages of building one device's dataset.

    1. make_unprocessed_data_files:
      pull each shot from the device's source (get_source_dataset),
      filter it down to the regions of validity (filter_and_plot),
      and write one netCDF plus one plot per shot.
    2. run_gp_fitting:
      map the Thomson channels onto rho_tor_norm (prepare_fit_input),
      stage method-agnostic batches,
      fit them here or on a cluster,
      and write one netCDF of fitted profiles per shot.
    3. stack_internal_dataset:
      stack every shot that has both into one Zarr store,
      with the slower profiles and equilibria held forward onto the
      1 kHz grid and flagged where they carry a sample of their own.
    4. publish_dataset:
      derive the published copy of the internal store, with
      published_strip_signals stripped out.

    Each stage resumes from what is already on disk, except the last two,
    which always rebuild their store. A device subclass supplies the source
    and the filter thresholds, everything else lives here.

    A shotlist may carry time windows per shot (see windows.read_shotlist).
    They leave the unprocessed stage alone and enter at fit staging, where
    only the Thomson samples inside them are fit, and at the stack
    stage, where the store is cut down to the grid times inside them.
    The fit_mode attribute names which of the three the run is in.
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
    def min_segment_length(self) -> float:
        """Minimum length of a single contiguous segment kept by the filters.

        Shorter segments are dropped, so that the sporadic few-millisecond chunks
        the filters leave behind do not reach the fits or the datasets.

        Returns:
            Minimum segment length in seconds.
        """

    @property
    @abstractmethod
    def shot_blacklist(self) -> list[int]:
        """List of shots to exclude from processing due to known issues.

        Returns:
            List of shot numbers to exclude.
        """

    # Threads used to stage source data.
    # Only above 1 for sources that tolerate concurrent reads
    # MAST reads public S3 and does, disruption-py's MDSplus connections do not.
    default_prepare_workers = 1

    # The device's settings dataclass, what the [<device>] table of the
    # config file is read into. Devices with settings override this.
    settings_cls: type[DeviceSettings] = DeviceSettings

    # Variables stripped from the published dataset.
    # The internal dataset keeps them.
    # Device subclasses override this to release more or hold back more.
    published_strip_signals: tuple[str, ...] = RAW_TS_CHANNEL_SIGNALS

    # Per-variable attributes the device owns (description, units, ref),
    # for the signals it renames or synthesizes rather than takes from the
    # source with attributes attached. Applied when the unprocessed file is
    # written and again at the stack stage, so the stores carry what the code says
    # now, not what the file said when it was pulled.
    signal_attrs: dict[str, dict] = {}

    # The process's log file sink, replaced when a new workflow starts.
    _log_sink_id = None

    # GP fit staging knobs
    fit_rho_tor_norm = np.linspace(0.0, 1.6, 81)
    fit_min_points = 10
    fit_scale_per_slice = False
    fit_bounds = default_fit_bounds()

    def __init__(
        self,
        ds_name: str,
        data_assembly_dir: Path,
        shotlist_file: Path | None = None,
        max_num_shots: int | None = None,
        average_windows: bool = False,
        fit_method: str = "zk",
        cluster_config: ClusterFitConfig | None = None,
        prepare_workers: int | None = None,
        settings: DeviceSettings | None = None,
    ):
        """Set up the dataset directories and resolve the shotlist.

        Args:
            ds_name: Name of the dataset.
            data_assembly_dir: Directory where data files are stored and the
                datasets will be saved.
            shotlist_file: Path to file containing list of shots to process,
                one shot per line, or a CSV with time windows per shot.
                If None, will call get_shotlist_from_source() to retrieve shotlist from
                device-specific source.
            max_num_shots: Maximum number of shots to process (for testing). If None,
                process all shots.
            average_windows: Pool every Thomson point inside a time window
                and fit each window as one profile, instead of fitting the
                Thomson samples inside it one by one. Needs a shotlist with windows.
            fit_method: GP fitting method name (see gp_fitting.registry). Names the
                fit result and fit plot subdirectories, and the cluster job's worker.
            cluster_config: If provided, GP profile fitting is dispatched to a SLURM
                cluster (see gp_fitting/dispatcher.py).
                If None, fitting runs single-threaded in this process.
            prepare_workers: Threads used to stage source data. None takes the
                device's default_prepare_workers.
            settings: The device's settings, an instance of its settings_cls
                (built from the config file by config.load_run_config). None
                takes the defaults.

        Raises:
            TypeError: If settings is not an instance of the device's settings_cls.
            ValueError: If average_windows is set without a windowed shotlist.
        """
        self.ds_name = ds_name
        self.data_assembly_dir = data_assembly_dir / self.ds_name

        self.max_num_shots = max_num_shots
        if max_num_shots is None:
            self.stores_dir = self.data_assembly_dir / "04_datasets"
        else:
            self.stores_dir = self.data_assembly_dir / f"04_datasets_{max_num_shots}"
        self.fit_method = fit_method
        self.cluster_config = cluster_config
        self.prepare_workers = (
            self.default_prepare_workers if prepare_workers is None else prepare_workers
        )
        self.settings = self.settings_cls() if settings is None else settings
        if not isinstance(self.settings, self.settings_cls):
            raise TypeError(
                f"{type(self).__name__} takes {self.settings_cls.__name__} settings, "
                f"got {type(self.settings).__name__}"
            )
        self.fit_anchors = _fit_anchors(self.settings)

        # Set up subdirectories for unprocessed data, fit staging, and the datasets
        self.unprocessed_data_dir = self.data_assembly_dir / "01_unprocessed"
        self.rejected_shots_dir = self.unprocessed_data_dir / "rejected_shots"
        self.accepted_shots_dir = self.unprocessed_data_dir / "accepted_shots"
        self.failed_shots_dir = self.unprocessed_data_dir / "failed_shots"
        self.fit_staging_dir = self.data_assembly_dir / "02_fit_staging"
        self.fit_batches_dir = self.fit_staging_dir / "batches"
        self.failed_fits_dir = self.fit_staging_dir / "failed_shots"
        self.fit_results_dir = self.data_assembly_dir / "03_fit_results"
        # One subdirectory per fitting method, so results of different methods
        # sit side by side rather than overwriting each other.
        self.fit_shots_dir = self.fit_results_dir / fit_method
        self.fit_plots_dir = self.fit_results_dir / "ts_fits" / fit_method
        # Optional; only touched by export_to_imas(), which needs the
        # `imas` extra (imas-python, eqdsk). One subdirectory
        # per shot, scoped by fit method like fit_shots_dir, since the
        # exported profiles/Zeff/impurity composition all derive from that
        # method's fit output.
        self.imas_export_dir = self.data_assembly_dir / "imas_export" / fit_method

        # Log the run to a timestamped file named for when it was launched.
        # One file sink per process: a later workflow in the same process
        # (tests, back-to-back builds) replaces the sink instead of adding a
        # second one, which would write both datasets' logs into both files.
        self.logs_dir = self.data_assembly_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        # The cluster fit jobs' SLURM logs, pulled back by the dispatcher
        self.fit_job_logs_dir = self.logs_dir / "fit_jobs"
        launch_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.logs_dir / f"{ds_name}_{launch_time}.log"
        if DataWorkflow._log_sink_id is not None:
            logger.remove(DataWorkflow._log_sink_id)
        DataWorkflow._log_sink_id = logger.add(self.log_file)
        logger.info(f"Logging this run to {self.log_file}")
        logger.info(f"Device settings: {self.settings}")

        if shotlist_file is None:
            logger.info(
                "No shotlist file provided, retrieving shotlist from device-specific source"
            )
            self.shotlist = self.get_shotlist_from_source()
            self.shot_windows = None
            logger.info(f"Retrieved {len(self.shotlist)} shots from source")
        else:
            self.shotlist, self.shot_windows = read_shotlist(shotlist_file)
            if self.shot_windows is None:
                logger.info(f"Loaded {len(self.shotlist)} shots from {shotlist_file}")
            else:
                n_windows = sum(len(w) for w in self.shot_windows.values())
                logger.info(
                    f"Loaded {len(self.shotlist)} shots with {n_windows} time windows "
                    f"from {shotlist_file}"
                )

        if average_windows and self.shot_windows is None:
            raise ValueError(
                "average_windows needs a shotlist with time windows (a CSV with t_start and t_end columns),"
                "there is nothing to average over without them"
            )
        if self.shot_windows is None:
            self.fit_mode = FIT_MODE_SAMPLE
        elif average_windows:
            self.fit_mode = FIT_MODE_WINDOW_AVERAGE
        else:
            self.fit_mode = FIT_MODE_WINDOW_SAMPLE
        logger.info(f"Fit mode: {self.fit_mode}")

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
    def get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from device-specific source.

        This method is called when no shotlist file is provided. Subclasses should
        implement their own logic (SQL database query, reading from existing dataset,
        etc.)

        Returns:
            Shot numbers to process.
        """

    @abstractmethod
    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        """Read one shot from the device's source into standardized signals.

        Everything the datasets need, under the standardized names, on
        the shot's uniform 1 kHz timebase. Filtering, plotting, and writing are
        make_unprocessed_data_files' job.

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
            A permanent problem (no source data, missing signals, no plasma)
            should also be recorded with record_failed_shot so later runs skip
            the shot; a transient read failure should only be logged, so that
            the next run retries it.
        """

    def make_unprocessed_data_files(self):
        """Pull data from source and filter down to regions of validity.

        Saves one file per shot in unprocessed_data_dir, with standardized
        signal names on a common 1 kHz timebase, plus a plot per shot showing
        what was kept and why.

        Resumes: shots that already have a file, are blacklisted, or failed on
        an earlier run are skipped without touching the source. Source reads
        run prepare_workers at a time; filtering, plotting, and writing stay on
        this thread, since they are matplotlib and netCDF work.
        """
        self.unprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        workers = max(1, self.prepare_workers or 1)
        shots = iter(self.shotlist)
        n_files = 0
        exhausted = False

        while not exhausted and (
            self.max_num_shots is None or n_files < self.max_num_shots
        ):
            # Fill a batch with shots that actually need a source read, so a
            # run that is mostly resuming does not stage one shot at a time
            batch: list[int] = []
            while len(batch) < workers:
                if (
                    self.max_num_shots is not None
                    and n_files + len(batch) >= self.max_num_shots
                ):
                    break
                shot = next(shots, None)
                if shot is None:
                    exhausted = True
                    break
                if shot in self.shot_blacklist:
                    logger.info(f"Shot {shot} is blacklisted. Skipping.")
                    continue
                if (self.unprocessed_data_dir / f"{shot}.nc").exists():
                    logger.info(
                        f"Unprocessed data file for shot {shot} already exists. Skipping."
                    )
                    n_files += 1
                    continue
                if self.shot_already_failed(shot):
                    logger.info(f"Shot {shot} failed on a previous run. Skipping.")
                    continue
                batch.append(shot)
            n_files += self._read_and_write_shots(batch, workers)

        logger.info(f"Finished with {n_files} {self.ds_name} unprocessed data files.")

    def _read_and_write_shots(self, shots: list[int], workers: int) -> int:
        """Read a batch of shots from the source, then filter and write them.

        Args:
            shots: Shot numbers to read.
            workers: Threads to read them with.

        Returns:
            How many unprocessed data files were written.
        """
        if not shots:
            return 0

        def read(shot: int) -> xr.Dataset | None:
            """Read one shot, keeping a failure from killing the whole run.

            Args:
                shot: Shot number to read.

            Returns:
                The standardized dataset, or None if it could not be read.
            """
            try:
                return self.get_source_dataset(shot)
            except Exception as e:
                logger.warning(f"Shot {shot}: failed to read source data: {e}")
                logger.opt(exception=True).debug(e)
                return None

        if len(shots) == 1 or workers == 1:
            datasets = [read(shot) for shot in shots]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                datasets = list(pool.map(read, shots))

        n_written = 0
        for shot, ds_standardized in zip(shots, datasets, strict=True):
            if ds_standardized is None:
                continue
            ds_unprocessed = self.filter_and_plot(ds_standardized)
            if ds_unprocessed is None:
                logger.warning(
                    f"Shot {shot} did not pass filtering. Skipping unprocessed data file creation."
                )
                self.record_failed_shot(shot, "Did not pass filtering.")
                continue
            ds_unprocessed = _clip_powers(ds_unprocessed)
            # What pulled the shot and what this package was when it did.
            # The source's own stamp is rewritten, see provenance.SOURCE_ATTR_KEYS.
            ds_unprocessed.attrs = {
                **source_provenance(ds_unprocessed.attrs),
                **build_provenance(),
            }
            ds_unprocessed.to_netcdf(self.unprocessed_data_dir / f"{shot}.nc")
            logger.info(f"Created unprocessed data file for shot {shot}.")
            n_written += 1
        return n_written

    def filter_and_plot(self, ds_input: xr.Dataset) -> xr.Dataset | None:
        """Take datasets with standardized names, run filtering on them, and plot results.

        What survives is whatever passes the valid and transient filters, minus the
        contiguous segments shorter than min_segment_length.

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

        # 3: Drop the short segments the filters leave behind.
        times = ds_input["time"].values
        time_mask = (
            valid_mask.any(dim="shot") if "shot" in valid_mask.dims else valid_mask
        )
        kept_mask, dropped_lengths = drop_short_segments(
            time_mask.values, times, self.min_segment_length
        )
        if dropped_lengths:
            logger.info(
                f"Shot {shot}: dropped {len(dropped_lengths)} segment(s) shorter than "
                f"{1e3 * self.min_segment_length:.0f} ms, lengths [ms]: "
                + ", ".join(f"{1e3 * length:.0f}" for length in dropped_lengths)
            )
        valid_mask = valid_mask & xr.DataArray(
            kept_mask, coords={"time": times}, dims="time"
        )

        # Load-bearing broadcast: valid_mask carries the shot and time dims, so
        # this also gives every static quantity (the limiter contour, the fixed
        # grid extents, C-Mod's fixed channel positions) a time axis.
        # The internal dataset carries them per slice like everything else, and
        # _hold_equilibrium indexes them by grid time.
        ds_filtered = ds_input.where(valid_mask, drop=True)

        # If there is not sufficient data after filtering, return None to indicate that this shot should be rejected:
        # the first and last non-nan ip must be at least min_pulse_length apart,
        # and the non-nan segments within must sum to at least min_usable_time
        ip_valid_filtered = ds_filtered["ip"].notnull().any(dim="shot")
        ip_times = ds_filtered["time"].values[ip_valid_filtered.values]
        pulse_length, usable_time = pulse_and_usable_time(ip_times)

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
            window_spans = (
                None if self.shot_windows is None else self.shot_windows.get(int(shot))
            )
            plot_unprocessed_data(
                ds_input,
                self.accepted_shots_dir / f"{shot}.png",
                title=f"Shot {shot}",
                valid_filter=self.valid_filter,
                transient_filter=self.transient_filter,
                end_margin_time=end_margin_time,
                transient_margin_time=transient_margin_time,
                kept_spans=list(zip(span_starts.tolist(), span_ends.tolist())),
                window_spans=window_spans,
            )
            return ds_filtered

    @abstractmethod
    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Build GP fit inputs for one shot from its unprocessed dataset.

        Subclasses map the TS channels onto rho_tor_norm, convert to the fit units
        (Te [keV], ne [1e20 m^-3]), and apply their device-specific channel
        quality screens and error floors.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset (one 01_unprocessed file).

        Returns:
            The fit input, or None when the shot has nothing fittable (the
            caller records it as failed).
        """

    def unprocessed_shots(self) -> list[int]:
        """List the shots that have unprocessed data files.

        Returns:
            Shot numbers with unprocessed data files in unprocessed_data_dir.
        """
        return sorted(int(p.stem) for p in self.unprocessed_data_dir.glob("*.nc"))

    def run_gp_fitting(self, max_pages: int | None = None):
        """Run GP profile fitting on the unprocessed data files.

        The stages, each skipping work that already exists on disk:
        1. Stage: map TS channels onto rho_tor_norm, apply device cleaning
           (prepare_fit_input) and the shared screens every method sees (cleaning.clean_fit_rows),
           restrict or pool the Thomson samples to the shotlist's
           time windows if it has any, and pack batch npz files into
           fit_staging_dir/batches. The staged batches are method-agnostic.
        2. Fit each batch with self.fit_method's worker: single-threaded in
           this process when cluster_config is None, otherwise dispatched to
           the SLURM cluster.
        3. Write the batch results out as one netCDF per shot into
           fit_shots_dir.
        4. Plot the fits per shot into fit_plots_dir.

        Args:
            max_pages: Maximum number of pages to plot per shot. None plots all.
        """
        shots = self.unprocessed_shots()
        skipped = {s for s in shots if self.fit_already_failed(s)}
        if skipped:
            logger.info(
                f"Skipping {len(skipped)} shots that failed fit staging on a previous run"
            )
        shots = [s for s in shots if s not in skipped]
        if self.shot_windows is not None:
            without_window = [s for s in shots if s not in self.shot_windows]
            if without_window:
                logger.info(
                    f"Skipping {len(without_window)} unprocessed shots that have no "
                    "time window in the shotlist"
                )
            shots = [s for s in shots if s in self.shot_windows]
        logger.info(
            f"GP fitting {len(shots)} shots with method '{self.fit_method}' "
            f"in fit mode '{self.fit_mode}'"
        )

        # Checked before any fitting, so a stale dataset fails before it costs anything
        self._check_staged_batches()

        batches = self.stage_fit_batches(shots)
        if self.cluster_config is None:
            self._fit_batches_local(batches)
        else:
            from transport_validation_datasets.gp_fitting.dispatcher import (
                ClusterFitDispatcher,
            )

            dispatcher = ClusterFitDispatcher(
                self.cluster_config,
                self.ds_name,
                self.fit_batches_dir,
                self.fit_method,
                self.fit_job_logs_dir,
            )
            dispatcher.run(batches)
        self.write_fit_results()
        self.plot_fit_results(max_pages=max_pages)

    def clean_fit_state(self):
        """Delete every staged fit batch so the next run refits from scratch.

        Destructive: fits already computed are lost, unprocessed data files are
        untouched. This method's fit result files and fit plots are deleted
        too: both are rebuilt from the new fits, and the plots would otherwise
        survive the rebuild (plot_fit_results skips shots whose PDF exists).
        With a cluster_config this also cancels the dataset's queued jobs and
        clears its remote batch files, which a from-scratch cluster fit needs:
        otherwise the next run adopts the cancelled jobs or pulls back the
        leftover results (see ClusterFitDispatcher.clean).
        """
        if self.cluster_config is not None:
            from transport_validation_datasets.gp_fitting.dispatcher import (
                ClusterFitDispatcher,
            )

            dispatcher = ClusterFitDispatcher(
                self.cluster_config,
                self.ds_name,
                self.fit_batches_dir,
                self.fit_method,
                self.fit_job_logs_dir,
            )
            dispatcher.clean()
        elif self.fit_batches_dir.exists():
            shutil.rmtree(self.fit_batches_dir)
        for out_dir in (self.fit_shots_dir, self.fit_plots_dir):
            if out_dir.exists():
                shutil.rmtree(out_dir)
        logger.info(f"Cleaned fit staging state in {self.fit_batches_dir}")

    def stage_fit_batches(self, shots: list[int]) -> dict[str, list[int]]:
        """Stage fit inputs for the given shots into batch npz files.

        Shots already covered by an existing batch file keep their batch
        (and with it the batch id a restarted cluster run lines up on),
        the rest are packed into new batches.
        A batch records the fit mode and the time windows it was staged with,
        and run_gp_fitting refuses the whole staging directory when any of it
        disagrees with this run (_check_staged_batches),
        so an edited shotlist never quietly reuses fits made for other windows.
        Every shot's per-sample rows go through cleaning.clean_fit_rows before any window pools them,
        and its staged rows through cleaning.drop_rows_without_core after.
        A shot whose prepare_fit_input returns None, or that cleaning leaves nothing fittable,
        is recorded as failed and skipped on later runs.
        A shot with no Thomson sample inside its windows is only logged, the windows may be different next run.

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
            self._batch_plan_name(), shots, self.fit_batches_dir, shots_per_batch
        )

        batches: dict[str, list[int]] = {}
        for batch_id, batch_shots in sorted(planned.items()):
            in_path = self._batch_in_path(batch_id)
            if in_path.exists():
                self._check_batch(in_path, batch_id)
                batches[batch_id] = batch_shots
                continue
            shot_inputs = {}
            for shot in batch_shots:
                with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds:
                    fit_input = self.prepare_fit_input(shot, ds)
                if fit_input is None:
                    self.record_failed_fit(shot, "No fittable Thomson channel data.")
                    continue
                # Per sample, before any window pools the samples, see cleaning.py
                fit_input = clean_fit_rows(fit_input, shot)
                if not fit_input.has_fittable_points():
                    self.record_failed_fit(
                        shot, "No fittable Thomson channel data after cleaning."
                    )
                    continue
                fit_input = self._apply_windows(shot, fit_input)
                if fit_input is None:
                    continue
                shot_inputs[shot] = drop_rows_without_core(fit_input, shot)
            if not shot_inputs:
                continue
            pack_fit_batch(
                in_path,
                FitBatch(
                    shot_inputs=shot_inputs,
                    x_star=self.fit_rho_tor_norm,
                    min_points=self.fit_min_points,
                    scale_per_slice=self.fit_scale_per_slice,
                    bounds=self.fit_bounds,
                    anchors=self.fit_anchors,
                    pedestal_rho_tor_norm=self.settings.pedestal_rho_tor_norm,
                    sol_extension=self.settings.sol_extension,
                    fit_mode=self.fit_mode,
                ),
            )
            batches[batch_id] = sorted(shot_inputs)
            logger.info(f"Staged batch {batch_id} with {len(shot_inputs)} shots")
        return batches

    def _batch_plan_name(self) -> str:
        """Name the batch ids are hashed from.

        The dataset name, salted with the fit mode when windows are in play,
        so a windowed run never lines up with a per-sample batch of the same
        shots, neither a local file nor a result or queued job on the
        cluster. Plain per-sample runs keep the ids they always had.

        Returns:
            The name to plan batches under.
        """
        if self.fit_mode == FIT_MODE_SAMPLE:
            return self.ds_name
        return f"{self.ds_name}:{self.fit_mode}"

    def _check_run_setting(self, name: str, found: str, expected: str, where: str):
        """Refuse something on disk built with another setting than this run's.

        Args:
            name: The setting, for the message.
            found: The value it records.
            expected: This run's value.
            where: What is being checked, for the message.

        Raises:
            ValueError: If the values differ.
        """
        if found != expected:
            raise ValueError(
                f"{where} was built with {name} {found}, this run has {expected}. "
                "Rerun with --clean_fit_state to rebuild it, "
                "or build the dataset under another ds_name."
            )

    def _check_batch(self, in_path: Path, batch_id: str):
        """Refuse a staged batch built with another fit mode, SOL extension, pedestal location, or anchors than this run's.

        Args:
            in_path: The batch input npz.
            batch_id: The batch id, for the message.
        """
        where = f"Batch {batch_id}"
        staged_mode = read_batch_setting(in_path, "fit_mode")
        self._check_run_setting("fit mode", staged_mode, self.fit_mode, where)
        staged_extension = read_batch_setting(in_path, "sol_extension")
        run_extension = self.settings.sol_extension
        self._check_run_setting("SOL extension", staged_extension, run_extension, where)
        # As floats, so a TOML integer matches the float the batch stores
        staged_pedestal_raw = read_batch_setting(in_path, "pedestal_rho_tor_norm")
        staged_pedestal = str(float(staged_pedestal_raw))
        run_pedestal = str(float(self.settings.pedestal_rho_tor_norm))
        self._check_run_setting(
            "pedestal location", staged_pedestal, run_pedestal, where
        )
        staged_anchors = read_batch_anchors(in_path)
        staged_anchors_json = _anchors_json(staged_anchors)
        run_anchors_json = _anchors_json(self.fit_anchors)
        self._check_run_setting("anchors", staged_anchors_json, run_anchors_json, where)

    def _check_staged_batches(self):
        """Refuse a staging directory that does not match this run.

        Every batch input in fit_batches_dir is checked,
        because write_fit_results and the stack stage sweep them all:
        its fit mode, SOL extension, pedestal location, and anchors must be this run's,
        and in a windowed run every shot must have the same windows it was staged with.
        An edited shotlist stops here, before any fit runs or any result is written.
        A batch in another mode, with another SOL extension, pedestal location, other anchors or windows,
        or holding a shot the shotlist no longer lists raises ValueError
        (_check_batch, _check_windows_match).
        """
        for in_path in sorted(self.fit_batches_dir.glob("batch_*.npz")):
            if "_out_" in in_path.name:
                continue
            batch_id = in_path.stem.removeprefix("batch_")
            self._check_batch(in_path, batch_id)
            if self.shot_windows is None:
                continue
            for shot, staged in read_batch_windows(in_path).items():
                self._check_windows_match(staged, shot, f"batch {batch_id}")

    def _check_windows_match(self, staged: np.ndarray, shot: int, where: str):
        """Refuse something on disk built with other windows than the shotlist's.

        Args:
            staged: (n_w, 2) windows it was built with, empty when it had none.
            shot: The shot it belongs to.
            where: What is being checked, for the message.

        Raises:
            ValueError: If the shotlist no longer lists the shot, or gives it
                other windows.
        """
        current = self.shot_windows.get(shot)
        staged = window_bounds(staged)
        if current is not None and np.array_equal(staged, window_bounds(current)):
            return
        if current is None:
            problem = "the shotlist no longer lists the shot"
        else:
            problem = (
                f"the shotlist now gives {np.asarray(current).tolist()}, it was "
                f"built with {staged.tolist()}"
            )
        raise ValueError(
            f"Shot {shot} in {where} was built with other time windows: {problem}. "
            f"Rerun with --clean_fit_state to restage everything, or build the "
            f"dataset under another ds_name."
        )

    def _apply_windows(self, shot: int, fit_input: ShotFitInput) -> ShotFitInput | None:
        """Restrict or pool one shot's per-sample fit input to its time windows.

        Args:
            shot: Shot number.
            fit_input: The shot's fit input as prepare_fit_input built it.

        Returns:
            The fit input as it is staged: untouched without windows, the
            rows inside the windows in per-sample mode, one pooled row per
            window in averaging mode. None when the shotlist gives the shot
            no window or no Thomson sample falls in one.
        """
        if self.shot_windows is None:
            return fit_input
        windows = self.shot_windows.get(shot)
        if windows is None:
            logger.warning(
                f"Shot {shot}: no time window in the shotlist, nothing to fit"
            )
            return None
        restricted = restrict_to_windows(fit_input, windows)
        if restricted is None:
            logger.warning(
                f"Shot {shot}: no Thomson sample inside its {len(windows)} time "
                "window(s), nothing to fit"
            )
            return None
        if self.fit_mode == FIT_MODE_WINDOW_AVERAGE:
            return pool_windows(restricted, windows, shot)
        return restricted

    def _batch_in_path(self, batch_id: str) -> Path:
        return self.fit_batches_dir / f"batch_{batch_id}.npz"

    def _batch_out_path(self, batch_id: str) -> Path:
        return self.fit_batches_dir / f"batch_{batch_id}_out_{self.fit_method}.npz"

    def _fit_batches_local(self, batches: dict[str, list[int]]):
        """Fit staged batches serially in this process, one at a time.

        Args:
            batches: Mapping of batch id to shots, from stage_fit_batches.
        """
        for i, batch_id in enumerate(sorted(batches)):
            out_path = self._batch_out_path(batch_id)
            if out_path.exists():
                continue
            logger.info(
                f"Fitting batch {batch_id} ({batches[batch_id]}) locally "
                f"({i + 1}/{len(batches)})"
            )
            registry.run_batch_file(
                self.fit_method, self._batch_in_path(batch_id), out_path
            )

    def write_fit_results(self):
        """Write every batch's fit results out as one netCDF per shot.

        One file per shot in fit_shots_dir, each written
        atomically. Nothing is ever held across batches, so this stays flat in
        memory no matter how many shots the dataset has.
        Profiles are converted back from the fit units to SI (Te [eV], ne [m^-3])
        to match the unprocessed files' conventions, gradients are per unit rho_tor_norm.
        Only the grid up to STORED_RHO_TOR_NORM_MAX is written.
        Batches without a result file yet are skipped with a warning, so a
        partially fit dataset still writes.
        A shot whose file is already newer than its batch result is left alone,
        so re-running after fitting a few more batches only writes those.
        Each file records the fit mode and, for a windowed run, the shot's
        windows and the window of every row.

        Raises:
            ValueError: If batches were fit on different rho_tor_norm grids, the
                staging directory does not match this run
                (_check_staged_batches), or a batch's result rows do not
                align with its staged input.
        """
        self._check_staged_batches()
        hyp_names = getattr(registry.load_worker(self.fit_method), "HYP_NAMES", None)
        x_star = None
        n_written = 0
        n_shots = 0
        for in_path in sorted(self.fit_batches_dir.glob("batch_*.npz")):
            if "_out_" in in_path.name:
                continue
            batch_id = in_path.stem.removeprefix("batch_")
            out_path = self._batch_out_path(batch_id)
            if not out_path.exists():
                logger.warning(
                    f"No {self.fit_method} results for batch {batch_id} yet, skipping its shots"
                )
                continue
            # Only the two small members are read here, not the profile arrays
            with np.load(out_path) as data:
                batch_x_star = data["x_star"]
                batch_shots = data["shots"].tolist()
            if x_star is None:
                x_star = batch_x_star
            elif not np.array_equal(x_star, batch_x_star):
                raise ValueError(
                    f"Batch {batch_id} was fit on a different rho_tor_norm grid, re-stage and refit"
                )
            n_shots += len(batch_shots)

            # Skip the batch entirely if every shot file already postdates it
            out_mtime = out_path.stat().st_mtime
            if all(
                (p := self.fit_shots_dir / f"{shot}.nc").exists()
                and p.stat().st_mtime >= out_mtime
                for shot in batch_shots
            ):
                continue

            batch = unpack_fit_batch(in_path)
            for shot, so in unpack_fit_results(out_path).items():
                if so.te_fit.shape != (so.time.size, batch_x_star.size):
                    raise ValueError(
                        f"Batch {batch_id} shot {shot}: result rows do not align with its slice times"
                    )
                si = batch.shot_inputs[shot]
                if si.time.size != so.time.size:
                    raise ValueError(
                        f"Batch {batch_id} shot {shot}: the staged input has "
                        f"{si.time.size} rows but the result {so.time.size}, they are "
                        f"out of step. Rerun with --clean_fit_state."
                    )
                ds = self._shot_fit_dataset(
                    shot, so, batch_x_star, hyp_names, si.windows, si.window_index
                )
                shot_path = self.fit_shots_dir / f"{shot}.nc"
                shot_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = shot_path.with_suffix(".nc.tmp")
                ds.to_netcdf(tmp_path)
                os.replace(tmp_path, shot_path)
                n_written += 1

        if n_shots == 0:
            logger.warning(f"No {self.fit_method} fit results to write")
        else:
            logger.info(
                f"Wrote {n_written} of {n_shots} {self.fit_method} shot fit files "
                f"to {self.fit_shots_dir} "
                f"({n_shots - n_written} already up to date)"
            )

    def _shot_fit_dataset(
        self,
        shot: int,
        so: ShotFitOutput,
        x_star: np.ndarray,
        hyp_names: list[str] | None,
        windows: np.ndarray,
        window_index: np.ndarray,
    ) -> xr.Dataset:
        """Build the one-shot dataset that write_fit_results saves.

        Carries a shot dimension of length 1 and an integer TIME_DIM index
        coordinate, enabling lining up shots whose slice counts differ
        without any of them being padded on disk.
        The shot's window list rides along as a JSON attribute and each row's
        window as a coordinate (empty and -1 without windows),
        so the stack stage can place the rows by window.
        The profiles are cut at STORED_RHO_TOR_NORM_MAX.

        Args:
            shot: Shot number.
            so: The shot's ShotFitOutput.
            x_star: (n_x,) rho_tor_norm grid the profiles were fit on.
            hyp_names: Names of the method's hyperparameters, None if it has none.
            windows: (n_w, 2) time windows the shot was staged with [s].
            window_index: (n_t,) window of each row, -1 without windows.

        Returns:
            The shot's fit results as a dataset.
        """
        status_attrs = {
            "description": "Per-slice fit status",
            "codes": ", ".join(f"{k}={v}" for k, v in STATUS_NAMES.items()),
        }
        # isclose keeps a grid point that float round-off puts just past the cut
        stored = (x_star <= STORED_RHO_TOR_NORM_MAX) | np.isclose(
            x_star, STORED_RHO_TOR_NORM_MAX
        )
        data_vars = {}
        has_hyps = False
        for var, si_factor, name, unit, desc in (
            ("te", 1.0e3, "t_e", "eV", "electron temperature"),
            ("ne", 1.0e20, "n_e", "m^-3", "electron density"),
        ):
            for suffix, out_suffix, extra in (
                ("fit", "", ""),
                ("std", "_error", "1-sigma predictive uncertainty of the "),
                ("grad", "_gradient", "d/drho_tor_norm gradient of the "),
                (
                    "grad_std",
                    "_gradient_error",
                    "1-sigma uncertainty of the d/drho_tor_norm gradient of the ",
                ),
            ):
                grad_unit = (
                    unit
                    if suffix in ("fit", "std")
                    else f"{unit} per unit rho_tor_norm"
                )
                fit_rows = getattr(so, f"{var}_{suffix}")
                profile = fit_rows[None, :, stored] * si_factor
                data_vars[f"{name}{out_suffix}"] = (
                    ("shot", TIME_DIM, "rho_tor_norm"),
                    profile.astype(np.float32),
                    {
                        "description": f"{extra}GP-fitted {desc} profile",
                        "units": grad_unit,
                    },
                )
            data_vars[f"{name}_fit_status"] = (
                ("shot", TIME_DIM),
                np.asarray(getattr(so, f"{var}_status"), dtype=np.int8)[None],
                status_attrs,
            )
            hyps = getattr(so, f"{var}_hyps")
            if hyps is not None:
                has_hyps = True
                data_vars[f"{name}_hyperparameters"] = (
                    ("shot", TIME_DIM, "hyperparameter"),
                    np.asarray(hyps, dtype=np.float32)[None],
                    {"description": f"Fitted GP hyperparameters of the {desc} fit"},
                )

        time_description = (
            "Time window centers"
            if self.fit_mode == FIT_MODE_WINDOW_AVERAGE
            else "Thomson slice times"
        )
        coords = {
            "shot": [shot],
            # Slice ordinal, not a physical coordinate
            TIME_DIM: np.arange(so.time.size),
            "rho_tor_norm": (
                "rho_tor_norm",
                x_star[stored],
                {"description": RHO_TOR_NORM_DEFINITION},
            ),
            TIME_COORD: (
                ("shot", TIME_DIM),
                np.asarray(so.time, dtype=np.float32)[None],
                {"units": "s", "description": time_description},
            ),
        }
        if has_hyps and hyp_names is not None:
            coords["hyperparameter"] = list(hyp_names)
        attrs = {
            "fit_method": self.fit_method,
            "fit_mode": self.fit_mode,
            "sol_extension": self.settings.sol_extension,
            "rho_tor_norm_definition": RHO_TOR_NORM_DEFINITION,
            "dataset_name": self.ds_name,
        }
        attrs["windows"] = json.dumps(window_bounds(windows).tolist())
        coords["window_index"] = (
            ("shot", TIME_DIM),
            np.asarray(window_index, dtype=np.int32)[None],
            {
                "description": "Index into the windows attribute of the time "
                "window this row belongs to, -1 without windows"
            },
        )

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    def fit_plot_channel_groups(self, shot: int) -> list | None:
        """Get the channel grouping used to color the fit plots.

        Subclasses can split channels by diagnostic
        (e.g. C-Mod core vs edge Thomson).
        The base implementation plots them as one group.

        Args:
            shot: Shot number being plotted.

        Returns:
            (mask, color, label) triples, or None for a single group.
        """
        return None

    def plot_fit_results(self, max_pages: int | None = None):
        """Plot the GP fits of every fitted shot, one PDF per shot.

        Plots the exact (cleaned, floored) channel data the fit consumed,
        straight from the staged batch files.
        The fits are drawn over the whole fit grid, past STORED_RHO_TOR_NORM_MAX, so the anchors show.
        Skips shots whose PDF already exists.
        A window-averaged fit gets one page per window, with every pooled point on it.

        Args:
            max_pages: Maximum number of pages to plot per shot. None plots all.
        """
        for in_path in sorted(self.fit_batches_dir.glob("batch_*.npz")):
            if "_out_" in in_path.name:
                continue
            batch_id = in_path.stem.removeprefix("batch_")
            out_path = self._batch_out_path(batch_id)
            if not out_path.exists():
                continue
            # Read the shot list first, so a batch whose plots are all on disk
            # never has its profile arrays unpacked
            if all(
                (self.fit_plots_dir / f"{shot}.pdf").exists()
                for shot in read_batch_shots(out_path)
            ):
                continue
            batch = None
            for shot, so in unpack_fit_results(out_path).items():
                pdf_path = self.fit_plots_dir / f"{shot}.pdf"
                if pdf_path.exists():
                    continue
                if batch is None:
                    batch = unpack_fit_batch(in_path)
                si = batch.shot_inputs[shot]
                window_bounds = (
                    si.windows[si.window_index]
                    if batch.fit_mode == FIT_MODE_WINDOW_AVERAGE
                    else None
                )
                n_pages = plot_ts_fits(
                    pdf_path,
                    shot,
                    ts_time=si.time,
                    rho_tor_norm_ch=si.x,
                    channel_data={
                        "te": (si.te_y, si.te_err),
                        "ne": (si.ne_y, si.ne_err),
                    },
                    fit_output=so,
                    rho_tor_norm_fit=batch.x_star,
                    channel_groups=_tile_channel_groups(
                        self.fit_plot_channel_groups(shot), si.x.shape[1]
                    ),
                    max_pages=max_pages,
                    window_bounds=window_bounds,
                )
                logger.info(f"Plotted {n_pages} fit pages for shot {shot}")

    def stack_internal_dataset(
        self,
        mb_per_chunk: int | None = 50,
        drop_unfit_slices: bool = True,
        forward_fill: bool = True,
        extend_existing: bool = False,
    ) -> Path:
        """Stack the unprocessed data and the fit results into the internal dataset.

        One tensorized Zarr store at stores_dir/<ds_name>_internal.zarr,
        holding every shot that has both an unprocessed data file and a fit
        result file. Shots are stacked along EPISODE_DIM and NaN padded along
        every other dimension, so shots of different lengths still line up.
        Only one shot is held in memory at a time.

        The store carries every signal needed for internal analysis.

        The timebase is the unprocessed data's uniform 1 kHz grid. TIME_DIM is
        the grid ordinal, so shots of different lengths can be padded to a
        common size, and the TIME_COORD variable carries the times themselves.
        The fitted profiles come one per Thomson sample and the equilibria on
        the reconstruction clock, (might be slower than 1 kHz), so they are held
        forward onto the grid with the fresh_profile and fresh_equilibrium
        flags marking the grid times that carry a recent sample.

        With a windowed shotlist only the grid times inside the windows are
        kept, and a window-averaged profile fills its whole window with
        fresh_profile marking the window center.

        Args:
            mb_per_chunk: Target size of a storage chunk, chunked along
                EPISODE_DIM only. None leaves the chunking alone.
            drop_unfit_slices: Ignore the slices whose te or ne fit did not come
                back usable (USABLE_FIT_STATUSES), as though the shot had no
                Thomson sample there. False places every slice on the grid,
                all-NaN profiles included.
            forward_fill: Hold each fitted profile and each equilibrium forward
                over the grid times that follow it, for at most
                MAX_HOLD_PERIODS of their own sampling period. False leaves the
                grid times between samples NaN.
            extend_existing: Append the shots that are not in the existing
                store yet, instead of replacing it. The shots already in it are
                left as they are, so this does not pick up refitted shots.

        Returns:
            Path of the Zarr store.

        Raises:
            ValueError: If no shot has both an unprocessed data file and a fit
                result file.
        """
        shots = self._internal_dataset_shots()
        if not shots:
            raise ValueError(
                f"No shots have both an unprocessed data file in {self.unprocessed_data_dir} "
                f"and a '{self.fit_method}' fit result file in {self.fit_shots_dir}"
            )
        logger.info(f"Stacking {len(shots)} shots into the internal dataset")

        zarr_path = self.stores_dir / f"{self.ds_name}_internal.zarr"
        if zarr_path.exists():
            if not extend_existing:
                logger.warning(
                    f"Replacing the existing internal dataset at {zarr_path}"
                )
                shutil.rmtree(zarr_path)
            else:
                with xr.open_zarr(zarr_path, consolidated=True) as ds_store:
                    in_store = {int(shot) for shot in ds_store[EPISODE_DIM].values}
                shots = [shot for shot in shots if shot not in in_store]
                logger.info(
                    f"{len(in_store)} shots are already in {zarr_path}, "
                    f"appending the {len(shots)} that are not"
                )
                if not shots:
                    return zarr_path

        # Bound every non-episode dimension up front so that no shot ever has to
        # extend the store, which would rewrite the chunks of every shot in it.
        dim_sizes = self._internal_dataset_dim_sizes(shots)
        logger.info(f"Internal dataset dimension bounds: {dim_sizes}")

        ds = build_tensorized_dataset(
            process_fn=lambda shot: self._internal_shot_dataset(
                shot, drop_unfit_slices, forward_fill
            ),
            identifiers=shots,
            zarr_path=zarr_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=extend_existing,
            mb_per_chunk=mb_per_chunk,
            dim_sizes=dim_sizes,
        )
        logger.info(
            f"Internal dataset at {zarr_path}: {dict(ds.sizes)}, "
            f"{len(ds.data_vars)} variables"
        )
        ds.close()
        self._write_store_provenance(zarr_path)
        return zarr_path

    def publish_dataset(self) -> Path:
        """Derive the published dataset from the internal one by stripping signals.

        The published store is the internal store minus published_strip_signals,
        with the dimensions only those variables used dropped along with them,
        coordinates included.

        Returns:
            Path of the published Zarr store,
            stores_dir/<ds_name>_published.zarr.

        Raises:
            ValueError: If there is no internal store to derive from.
        """
        internal_path = self.stores_dir / f"{self.ds_name}_internal.zarr"
        published_path = self.stores_dir / f"{self.ds_name}_published.zarr"
        if not internal_path.exists():
            raise ValueError(
                f"No internal dataset at {internal_path}, run stack_internal_dataset first"
            )

        ds_internal = xr.open_zarr(internal_path, consolidated=True)
        stripped = [
            name
            for name in self.published_strip_signals
            if name in ds_internal.data_vars
        ]
        absent = sorted(set(self.published_strip_signals) - set(stripped))
        if absent:
            logger.warning(
                f"Signals to strip that the internal dataset does not carry: {absent}"
            )
        ds_published = ds_internal.drop_vars(stripped)
        used_dims = {
            dim for variable in ds_published.data_vars.values() for dim in variable.dims
        }
        ds_published = ds_published.drop_dims(
            [dim for dim in ds_published.dims if dim not in used_dims]
        )
        ds_published.attrs["stripped_signals"] = (
            ", ".join(stripped) if stripped else "none"
        )

        if published_path.exists():
            logger.warning(f"Replacing the published dataset at {published_path}")
            shutil.rmtree(published_path)
        ds_published.to_zarr(published_path, mode="w", consolidated=True)
        ds_internal.close()
        logger.info(
            f"Published dataset at {published_path}: "
            f"{len(ds_published.data_vars)} variables, {len(stripped)} stripped"
        )
        return published_path

    def _internal_dataset_shots(self) -> list[int]:
        """List the shots that can go into the internal dataset.

        Returns:
            Sorted shots that have both an unprocessed data file and a fit
            result file.
        """
        unprocessed = {int(p.stem) for p in self.unprocessed_data_dir.glob("*.nc")}
        fitted = {int(p.stem) for p in self.fit_shots_dir.glob("*.nc")}
        missing = unprocessed - fitted
        if missing:
            logger.warning(
                f"{len(missing)} unprocessed shots have no '{self.fit_method}' fit results yet "
                f"and are left out of the internal dataset"
            )
        return sorted(unprocessed & fitted)

    def export_to_imas(self, overwrite: bool = False):
        """Writes every fitted shot's equilibrium/core_profiles/summary/wall to IMAS format.

        Optional post-fitting step, requiring the `imas` extra (imas-python,
        eqdsk) -- the only stage that does; the rest of this
        package works without it. One shot-scoped output directory per shot
        under `imas_export_dir`, holding `equilibrium.nc`/`core_profiles.nc`/
        `summary.nc`/`wall.nc` plus the per-equilibrium-time `.geqdsk` files
        the equilibrium IDS was built from.

        `core_profiles` here carries electrons + a single hydrogenic main ion
        only (Zeff=1, n_D=n_e).

        Args:
            overwrite: Rewrite a shot's IMAS output even if it already exists.
        """
        from transport_validation_datasets.imas_export.scenario_export import (
            build_imas_from_shot,
            write_ids,
        )

        shots = self._final_dataset_shots()
        if not shots:
            logger.warning(
                "No shots have both unprocessed data and fit results; nothing to export to IMAS"
            )
            return

        n_written = 0
        for shot in shots:
            shot_dir = self.imas_export_dir / str(shot)
            if (shot_dir / "core_profiles.nc").exists() and not overwrite:
                continue
            fit_ds = xr.open_dataset(self.fit_shots_dir / f"{shot}.nc")
            unprocessed_ds = xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc")
            try:
                ids_list = build_imas_from_shot(
                    shot,
                    fit_ds,
                    unprocessed_ds,
                    geqdsk_dir=shot_dir / "geqdsk",
                )
            except Exception as e:
                logger.warning(
                    f"IMAS export failed for shot {shot}: {type(e).__name__}: {e}"
                )
                continue
            for ids in ids_list:
                write_ids(ids, shot_dir, overwrite=True)
            n_written += 1

        logger.info(
            f"Wrote IMAS output for {n_written} of {len(shots)} shots to {self.imas_export_dir} "
            f"({len(shots) - n_written} already up to date or failed)"
        )

    def _internal_dataset_dim_sizes(self, shots: list[int]) -> dict[str, int]:
        """Find the largest size of every non-episode dimension over the shots.

        Reads only the headers of the files (and the time coordinate of a windowed shot),
        so this stays cheap no matter how many shots the dataset has.
        Also the place every fit file is checked against this run, up front,
        since an error raised while a shot is being stacked would only skip that shot:
        its fit mode and SOL extension must be this run's and, in a windowed run,
        its windows the shotlist's, else ValueError (_check_run_setting, _check_windows_match).

        Args:
            shots: Shots that will go into the internal dataset.

        Returns:
            Upper bound per non-episode dimension.
        """
        sizes: dict[str, int] = {}
        seen: dict[str, set[int]] = {}
        for shot in shots:
            # The fit results only set the profile dimensions, their slices are
            # placed on the unprocessed timebase rather than kept as one
            with xr.open_dataset(self.fit_shots_dir / f"{shot}.nc") as ds_fit:
                self._check_run_setting(
                    "fit mode",
                    ds_fit.attrs["fit_mode"],
                    self.fit_mode,
                    f"The fit result file of shot {shot}",
                )
                self._check_run_setting(
                    "SOL extension",
                    ds_fit.attrs["sol_extension"],
                    self.settings.sol_extension,
                    f"The fit result file of shot {shot}",
                )
                windows = _fit_windows(ds_fit)
                if self.shot_windows is not None:
                    self._check_windows_match(windows, shot, "its fit results")
                shot_sizes = {
                    dim: size
                    for dim, size in ds_fit.sizes.items()
                    if dim not in (EPISODE_DIM, TIME_DIM, "hyperparameter")
                }
            with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds_shot:
                shot_sizes.update(
                    {
                        (TIME_DIM if dim == TIME_COORD else dim): size
                        for dim, size in ds_shot.sizes.items()
                        if dim != EPISODE_DIM
                    }
                )
                if windows.size:
                    grid = np.asarray(ds_shot[TIME_COORD].values, dtype=float)
                    shot_sizes[TIME_DIM] = int(in_any_window(grid, windows).sum())
            for dim, size in shot_sizes.items():
                sizes[dim] = max(sizes.get(dim, 0), size)
                seen.setdefault(dim, set()).add(size)

        varying = {
            dim: sorted(values) for dim, values in seen.items() if len(values) > 1
        }
        # A varying TIME_DIM is normal (shots run for different lengths), the
        # rest are grids that are supposed to be fixed for the device: their
        # coordinate values come from the first shot, so flag the mismatch.
        varying.pop(TIME_DIM, None)
        if varying:
            logger.warning(
                f"Dimensions differ between shots: {varying}. The internal dataset takes "
                f"their coordinate values from the first shot and NaN pads the rest."
            )
        return sizes

    @staticmethod
    def _usable_slice_mask(ds_fit: xr.Dataset) -> np.ndarray:
        """Find the slices whose te and ne fits both came back usable.

        Args:
            ds_fit: One shot's fit result dataset.

        Returns:
            (n_t,) boolean mask over the shot's slices.
        """
        mask = np.ones(ds_fit.sizes[TIME_DIM], dtype=bool)
        for name in ("t_e_fit_status", "n_e_fit_status"):
            status = ds_fit[name].squeeze(EPISODE_DIM, drop=True).values
            mask &= np.isin(status, USABLE_FIT_STATUSES)
        return mask

    def _internal_shot_dataset(
        self, shot: int, drop_unfit_slices: bool, forward_fill: bool
    ) -> xr.Dataset | None:
        """Build one shot's contribution to the internal dataset.

        Internal dataset is on a 1 kHz timebase from the unprocessed dataset.
        The fitted profiles are sampled far more slowly than that (one per Thomson sample).
        In addition, on some devices the equilibria are sampled more slowly
        and on some devices the equilibrium is too, so both are held forward
        onto the grid and the fresh flags mark which grid times carry a sample
        of their own.

        A shot fit with time windows is placed on its full grid first, then
        cut down to the grid times inside the windows, so an equilibrium from
        just before a window still fills its first milliseconds.
        Per-sample fits are never held across a window boundary.
        A window-averaged profile fills its whole window, fresh at the window center.

        Args:
            shot: Shot number.
            drop_unfit_slices: Ignore the slices whose te or ne fit did not come
                back usable, as though the shot had no Thomson sample there.
            forward_fill: Hold profiles and equilibria forward onto the grid
                times between their samples.

        Returns:
            The shot's dataset, or None when it has no usable fitted slice.

        Raises:
            ValueError: If the fitted slice times are not on the unprocessed
                file's timebase, or no time window overlaps it.
        """
        with xr.open_dataset(self.fit_shots_dir / f"{shot}.nc") as ds_file:
            ds_fit = ds_file.load()
        fit_mode = ds_fit.attrs["fit_mode"]
        # The full window list, before the unusable rows go:
        # a window whose every fit was culled still bounds the stored grid
        windows = _fit_windows(ds_fit)
        keep = (
            self._usable_slice_mask(ds_fit)
            if drop_unfit_slices
            else np.ones(ds_fit.sizes[TIME_DIM], dtype=bool)
        )
        if not keep.any():
            logger.warning(
                f"Shot {shot}: no usable {self.fit_method} fits, leaving it out of the internal dataset"
            )
            return None
        if not keep.all():
            logger.debug(
                f"Shot {shot}: keeping {int(keep.sum())} of {keep.size} fitted slices"
            )
        ds_fit = ds_fit.isel({TIME_DIM: np.flatnonzero(keep)})
        # A per-method fit diagnostic, kept in the fit files but left out of
        # the store, where it would be the only string-labelled dimension
        if "hyperparameter" in ds_fit.dims:
            ds_fit = ds_fit.drop_dims("hyperparameter", errors="ignore")
        slice_times = np.asarray(
            ds_fit[TIME_COORD].squeeze(EPISODE_DIM, drop=True).values, dtype=float
        )
        window_index = np.asarray(
            ds_fit["window_index"].squeeze(EPISODE_DIM, drop=True).values, dtype=int
        )
        # The grid time is the time of every row now, so the slice times only
        # come through as the fresh_profile flag, and the windows only place the rows
        ds_fit = ds_fit.drop_vars([TIME_COORD, "window_index"])

        with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds_file:
            ds_unprocessed = ds_file.load()
        grid = np.asarray(ds_unprocessed[TIME_COORD].values, dtype=float)
        # Dropped so the dimension can be reindexed and renamed,
        # the times come back as a per-shot variable at the end
        ds_unprocessed = ds_unprocessed.drop_vars(TIME_COORD)

        if fit_mode == FIT_MODE_WINDOW_AVERAGE:
            slice_index, fresh_profile = _place_windows_on_grid(
                grid, windows, window_index, forward_fill, shot
            )
            if not fresh_profile.any():
                raise ValueError(
                    f"Shot {shot}: none of its {windows.shape[0]} time windows overlaps "
                    f"the unprocessed file's timebase. The fit results and the "
                    f"unprocessed file are out of step, refit the shot."
                )
        else:
            # The fits of a windowed run only cover the Thomson samples inside the
            # windows, so the hold period comes from every sample in the file,
            # not from the fitted slice times with the inter-window gaps in them
            period = (
                _sample_period(ds_unprocessed, grid)
                if fit_mode == FIT_MODE_WINDOW_SAMPLE
                else None
            )
            slice_index, fresh_profile = _hold_onto_grid(
                grid, slice_times, forward_fill, period
            )
            if not fresh_profile.any():
                raise ValueError(
                    f"Shot {shot}: none of the {slice_times.size} fitted slice times are on "
                    f"the unprocessed file's timebase. The fit results and the unprocessed "
                    f"file are out of step, refit the shot."
                )
            if windows.size:
                # A hold stops where the grid time and the sample no longer share a window
                held = slice_index >= 0
                shared = (
                    window_membership(grid, windows)
                    & window_membership(slice_times, windows)[
                        np.clip(slice_index, 0, None)
                    ]
                ).any(axis=1)
                slice_index[held & ~shared] = -1

        data_vars = {}
        missing = []
        for name in DATASET_0D_SIGNALS:
            if name in ds_unprocessed:
                data_vars[name] = ds_unprocessed[name]
            else:
                # Kept as NaN so every device's dataset has one schema
                missing.append(name)
                data_vars[name] = xr.DataArray(
                    np.full((1, grid.size), np.nan, dtype=np.float32),
                    dims=(EPISODE_DIM, TIME_DIM),
                    attrs={"description": f"{name}, not available on this device"},
                )
        if missing:
            logger.debug(f"Shot {shot}: filling {missing} with NaN, not on this device")

        equilibrium, fresh_equilibrium = _hold_equilibrium(
            ds_unprocessed, grid, forward_fill
        )
        data_vars.update(equilibrium)

        # Every other unprocessed signal (the raw Thomson channels) comes
        # through as it sits on the grid. Internal-only ones are stripped from
        # the published store, not here (publish_dataset).
        for name in ds_unprocessed.data_vars:
            if name not in data_vars:
                data_vars[name] = ds_unprocessed[name]

        # Grid times are NaN padded up to the longest shot in the store, so the
        # integer status codes have to be floats to carry the padding
        for name in ("t_e_fit_status", "n_e_fit_status"):
            if name in ds_fit:
                ds_fit[name] = ds_fit[name].astype(np.float32)
        ds_fit = _hold_fits_onto_grid(ds_fit, slice_index)

        ds_grid = xr.Dataset(data_vars).rename({TIME_COORD: TIME_DIM})
        ordinal = np.arange(grid.size)
        # drop_conflicts, not drop: the latter also drops the per-variable attrs
        ds_stacked = xr.merge(
            [
                ds_grid.assign_coords({TIME_DIM: ordinal}),
                ds_fit.assign_coords({TIME_DIM: ordinal}),
                self._time_variables(grid, fresh_profile, fresh_equilibrium, fit_mode),
            ],
            combine_attrs="drop_conflicts",
        )
        # Non-index coordinates would become per-shot variables in the store,
        # and string-valued ones cannot be NaN padded
        extra_coords = [
            name
            for name in ds_stacked.coords
            if name not in ds_stacked.dims and name != TIME_COORD
        ]
        ds_stacked = ds_stacked.drop_vars(extra_coords)
        ds_stacked = ds_stacked.transpose(EPISODE_DIM, TIME_DIM, ...)

        # One dtype across the devices, whatever their staging wrote: the fits
        # are float32 already and the unprocessed files stay the full precision
        # source. Halves the store, which psirz dominates.
        for name, variable in ds_stacked.data_vars.items():
            if variable.dtype == np.float64:
                ds_stacked[name] = variable.astype(np.float32)
        for name, attrs in self.signal_attrs.items():
            if name in ds_stacked:
                ds_stacked[name].attrs.update(attrs)
        standardize_signal_attrs(ds_stacked)

        # The sign convention of the GEQDSK block, per shot since a dataset
        # may mix field directions: the unprocessed file's attribute, or
        # inferred from the signs the way the devices set it, for files
        # from before the attribute was kept
        cocos = ds_unprocessed.attrs.get("cocos")
        if cocos is None and "current" in ds_unprocessed and "bcentr" in ds_unprocessed:
            cocos = efit_cocos_from_signs(
                ds_unprocessed["current"].values, ds_unprocessed["bcentr"].values
            )
        ds_stacked["cocos"] = xr.DataArray(
            np.array([np.nan if cocos is None else cocos], dtype=np.float32),
            dims=(EPISODE_DIM,),
            attrs={
                "description": "COCOS index of the GEQDSK equilibrium signals "
                "(psirz, fpol, qpsi, current, ...), see "
                "https://doi.org/10.1016/j.cpc.2012.09.010. "
                "NaN when the shot has no equilibrium."
            },
        )

        time_definition = (
            "The unprocessed data's uniform 1 kHz timebase [s]. Profiles and "
            "equilibria are sampled more slowly and are held forward onto it, "
            "see fresh_profile and fresh_equilibrium."
        )
        if windows.size:
            # Cut last, so the holds above saw the full grid
            in_window = np.flatnonzero(in_any_window(grid, windows))
            ds_stacked = ds_stacked.isel({TIME_DIM: in_window}).assign_coords(
                {TIME_DIM: np.arange(in_window.size)}
            )
            time_definition += (
                " Only the grid times inside the shotlist's time windows are kept."
            )
        if fit_mode == FIT_MODE_WINDOW_AVERAGE:
            time_definition += (
                " Each profile is the fit of every Thomson point in its window "
                "and fills the whole window."
            )

        # Only what holds for the whole run.
        # The shots' own provenance is merged over every shot in the store once it is built
        ds_stacked.attrs = {
            "dataset_name": self.ds_name,
            "fit_method": self.fit_method,
            "fit_mode": fit_mode,
            "sol_extension": ds_fit.attrs["sol_extension"],
            "rho_tor_norm_definition": RHO_TOR_NORM_DEFINITION,
            "time_definition": time_definition,
        }
        return ds_stacked

    def _write_store_provenance(self, zarr_path: Path):
        """Add the run's provenance to the root attributes of a finished store.

        Three layers on top of the run attributes _internal_shot_dataset set:
        the shots' own provenance (the source package and version that pulled
        each unprocessed file, this package's commit when it did), merged over
        every shot in the store so a key the shots disagree on is recorded as
        the list of its values.
        Composed of this build's stamp and this package's state now
        (build_stamp, build_provenance), and the run configuration (_run_config_attrs).
        Unprocessed files from before the source stamp was rewritten are rewritten here,
        so the store never carries disruption-py's working-directory commit.

        Args:
            zarr_path: The internal store, complete and consolidated.
        """
        with xr.open_zarr(zarr_path, consolidated=True) as ds_store:
            shots = [int(shot) for shot in ds_store[EPISODE_DIM].values]
            attrs = dict(ds_store.attrs)
        per_shot = []
        for shot in shots:
            with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds_file:
                per_shot.append(source_provenance(ds_file.attrs))
        # cocos is per shot, the store carries it as a variable
        attrs.update(
            merge_shot_attrs(per_shot, exclude=(*SOURCE_VOLATILE_KEYS, "cocos"))
        )
        attrs.update(build_stamp())
        attrs.update(build_provenance())
        attrs.update(self._run_config_attrs())
        zarr.open_group(zarr_path, mode="r+").attrs.put(attrs)
        zarr.consolidate_metadata(zarr_path)

    def _run_config_attrs(self) -> dict[str, str]:
        """The choices this run was built with, as JSON attributes.

        Returns:
            device_settings (the settings_cls instance), filters (every
            threshold the unprocessed stage applied), and fit_settings (the
            staging knobs the fits were made with,
            including the full fit grid the store's rho_tor_norm coordinate is cut from).
        """
        return {
            "device_settings": to_json(self.settings),
            "filters": to_json(
                {
                    "valid_filter": self.valid_filter,
                    "transient_filter": self.transient_filter,
                    "transient_smoothing_window": TRANSIENT_SMOOTHING_WINDOW,
                    "end_margin": self.end_margin,
                    "min_pulse_length": self.min_pulse_length,
                    "min_usable_time": self.min_usable_time,
                    "min_segment_length": self.min_segment_length,
                    "shot_blacklist": list(self.shot_blacklist),
                }
            ),
            "fit_settings": to_json(
                {
                    "rho_tor_norm_grid": self.fit_rho_tor_norm,
                    "min_points": self.fit_min_points,
                    "scale_per_slice": self.fit_scale_per_slice,
                    "bounds": self.fit_bounds,
                    "max_hold_periods": MAX_HOLD_PERIODS,
                }
            ),
        }

    @staticmethod
    def _time_variables(
        grid: np.ndarray,
        fresh_profile: np.ndarray,
        fresh_equilibrium: np.ndarray,
        fit_mode: str = FIT_MODE_SAMPLE,
    ) -> xr.Dataset:
        """Build the per-shot time variable and the freshness flags.

        All three are floats because the store NaN pads them out to the longest
        shot in it, which an integer flag could not carry.

        Args:
            grid: The shot's 1 kHz timebase [s].
            fresh_profile: Grid times carrying a Thomson slice of their own,
                or the window centers of a window-averaged shot.
            fresh_equilibrium: Grid times carrying a reconstruction of their own.
            fit_mode: How the profiles were fit, one of the FIT_MODE_* values.
                Only changes what fresh_profile is described as.

        Returns:
            Dataset of the three variables, on (EPISODE_DIM, TIME_DIM).
        """
        held = (
            "1 where this grid time carries its own {}, 0 where it holds an earlier one"
        )
        profile_description = (
            "1 at the center grid time of each averaging window, 0 across the rest "
            "of the window, which holds the same window-averaged profile"
            if fit_mode == FIT_MODE_WINDOW_AVERAGE
            else held.format("Thomson slice")
        )
        return xr.Dataset(
            {
                TIME_COORD: (
                    (EPISODE_DIM, TIME_DIM),
                    grid[None].astype(np.float32),
                    {"units": "s", "description": "Uniform 1 kHz timebase"},
                ),
                "fresh_profile": (
                    (EPISODE_DIM, TIME_DIM),
                    fresh_profile[None].astype(np.float32),
                    {"description": profile_description},
                ),
                "fresh_equilibrium": (
                    (EPISODE_DIM, TIME_DIM),
                    fresh_equilibrium[None].astype(np.float32),
                    {"description": held.format("equilibrium reconstruction")},
                ),
            }
        )


def _clip_powers(ds: xr.Dataset) -> xr.Dataset:
    """Clip every power signal at zero before the unprocessed file is written.

    Source power records dip negative (bolometer baseline drift on cmod's
    power_radiated, ICRF pickup, MAST's power_nbi baseline), and no heating or
    radiated power is physically negative. Runs after filtering so the validity
    and transient gates still judge the values the device recorded.

    Args:
        ds: One shot's filtered dataset with standardized names.

    Returns:
        The same dataset with power signals clipped to >= 0, NaN untouched.
    """
    for name in DATASET_0D_SIGNALS:
        if name.startswith("power_") and name in ds:
            attrs = ds[name].attrs
            ds[name] = ds[name].clip(min=0.0) + 0.0  # -0.0 -> 0.0
            ds[name].attrs = attrs
    return ds


def drop_short_segments(
    keep: np.ndarray, times: np.ndarray, min_length: float
) -> tuple[np.ndarray, list[float]]:
    """Clear the runs of kept samples that are shorter than min_length.

    A segment is measured from its first to its last sample, the same way the
    pulse length and the plotted spans are, so a segment of n samples on the
    1 kHz grid is n - 1 milliseconds long.

    Args:
        keep: Mask over times, True where the sample survived the filters.
        times: The shot's timebase in seconds.
        min_length: Shortest segment to keep, in seconds.

    Returns:
        The mask with the short runs cleared, and the lengths in seconds of
        the runs that were cleared.
    """
    keep = np.asarray(keep, dtype=bool).copy()
    # Pad with False on both sides so a run touching either end still has an edge
    edges = np.diff(np.concatenate(([False], keep, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1  # Inclusive

    dropped_lengths = []
    for start, end in zip(starts, ends):
        length = float(times[end] - times[start])
        if length < min_length:
            keep[start : end + 1] = False
            dropped_lengths.append(length)
    return keep, dropped_lengths


def pulse_and_usable_time(kept_times: np.ndarray) -> tuple[float, float]:
    """Measure what the filters left of a shot, for the min_pulse_length and min_usable_time gates.

    Args:
        kept_times: The kept times on the uniform 1 kHz grid [s].

    Returns:
        (pulse_length, usable_time): the time from the first to the last kept sample,
        and the summed length of the kept segments [s].
    """
    if kept_times.size < 2:
        return 0.0, 0.0
    pulse_length = float(kept_times[-1] - kept_times[0])
    # Timebase is uniform 1 kHz, so any gap beyond 1.5 ms separates two segments
    gaps = np.diff(kept_times)
    usable_time = float(gaps[gaps < 1.5e-3].sum())
    return pulse_length, usable_time


def _hold_onto_grid(
    grid: np.ndarray,
    sample_times: np.ndarray,
    forward_fill: bool,
    period: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every grid time onto the sample it takes its values from.

    A grid time that carries a sample of its own takes that one. The rest
    take the most recent earlier sample, held for at most MAX_HOLD_PERIODS
    sampling periods so that nothing is carried across a long gap: the end
    of the shot, a diagnostic dropping out, or a stretch the filtering cut
    away. With forward_fill False nothing is held and only the grid times
    that carry a sample of their own come out.

    Args:
        grid: The shot's 1 kHz timebase [s].
        sample_times: Times of the samples to place on it [s], ascending.
        forward_fill: Hold each sample forward until the next one.
        period: The sampling period to hold for [s]. None takes the median
            spacing of sample_times, which is right when they are every
            sample there is, and wrong when they are a windowed subset.

    Returns:
        (sample_index, fresh): sample_index[i] is the index of the sample
        that grid time i draws from, -1 where it draws from none, and
        fresh[i] marks the grid times that carry a sample of their own.
    """
    sample_index = np.full(grid.size, -1, dtype=int)
    fresh = np.zeros(grid.size, dtype=bool)
    if sample_times.size == 0:
        return sample_index, fresh

    # Index of the last sample at or before each grid time
    previous_sample = (
        np.searchsorted(sample_times, grid + SAMPLE_TIME_TOL, side="right") - 1
    )
    has_previous = previous_sample >= 0
    age = grid - sample_times[np.clip(previous_sample, 0, None)]
    fresh = has_previous & (np.abs(age) <= SAMPLE_TIME_TOL)
    if not forward_fill:
        sample_index[fresh] = previous_sample[fresh]
        return sample_index, fresh

    # One sample on its own has no period to hold for,
    # so it only fills the grid step it sits on
    if period is None:
        period = np.median(
            np.diff(sample_times) if sample_times.size > 1 else np.diff(grid)
        )
    still_held = has_previous & (age <= MAX_HOLD_PERIODS * float(period))
    sample_index[still_held] = previous_sample[still_held]
    return sample_index, fresh


def _hold_fits_onto_grid(ds_fit: xr.Dataset, slice_index: np.ndarray) -> xr.Dataset:
    """Place a shot's fit results on its 1 kHz grid.

    Args:
        ds_fit: The shot's fit results, one row per Thomson slice.
        slice_index: Slice each grid time draws from, from _hold_onto_grid.

    Returns:
        The fit results on the grid, NaN at the grid times that draw on no
        slice.
    """
    ds_grid = ds_fit.isel({TIME_DIM: np.clip(slice_index, 0, None)})
    return ds_grid.where(xr.DataArray(slice_index >= 0, dims=TIME_DIM))


def _fit_windows(ds_fit: xr.Dataset) -> np.ndarray:
    """Read the time windows a shot's fit results were staged with.

    Args:
        ds_fit: One shot's fit result dataset.

    Returns:
        (n_w, 2) window bounds [s] sorted by start, empty for a shot fit
        without windows.
    """
    return window_bounds(json.loads(ds_fit.attrs["windows"]))


def _sample_period(ds_unprocessed: xr.Dataset, grid: np.ndarray) -> float | None:
    """Median spacing of a shot's Thomson samples, from its unprocessed file.

    Args:
        ds_unprocessed: The shot's unprocessed dataset, on the grid, with
            its time coordinate already dropped.
        grid: The shot's 1 kHz timebase [s].

    Returns:
        The period [s], or None when the file holds fewer than two samples
        (the caller then falls back to _hold_onto_grid's own estimate).
    """
    if "ts_channel_t_e" not in ds_unprocessed or "ts_channel_n_e" not in ds_unprocessed:
        return None
    has_sample = (
        (
            ds_unprocessed["ts_channel_t_e"].notnull()
            | ds_unprocessed["ts_channel_n_e"].notnull()
        )
        .any(dim="ts_channel")
        .squeeze(EPISODE_DIM, drop=True)
        .transpose(TIME_COORD)
        .values
    )
    sample_times = grid[has_sample]
    if sample_times.size < 2:
        return None
    return float(np.median(np.diff(sample_times)))


def _place_windows_on_grid(
    grid: np.ndarray,
    windows: np.ndarray,
    window_index: np.ndarray,
    forward_fill: bool,
    shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every grid time onto the window-averaged profile it takes its values from.

    A pooled profile stands for its whole window, so every grid time inside
    the window draws from it, and the one nearest the window center is
    flagged fresh. Where windows overlap, a grid time draws from the holding
    window whose center is nearest (the earlier one on a tie).
    With forward_fill False only the center times draw.
    A window with no grid time left in the unprocessed file (the filtering cut it away)
    is logged and its profile left out.

    Args:
        grid: The shot's 1 kHz timebase [s].
        windows: (n_w, 2) window bounds [s], sorted by start.
        window_index: (n_rows,) window of each kept fit row.
        forward_fill: Fill the whole window, not only its center.
        shot: Shot number, for the log lines.

    Returns:
        (sample_index, fresh) as _hold_onto_grid returns them: the fit row
        each grid time draws from, -1 for none, and the window centers.
    """
    sample_index = np.full(grid.size, -1, dtype=int)
    fresh = np.zeros(grid.size, dtype=bool)
    window_index = np.asarray(window_index, dtype=int)
    if window_index.size == 0:
        return sample_index, fresh
    windows = window_bounds(windows)
    # (n_grid, n_rows): which fitted rows' windows hold each grid time
    member = window_membership(grid, windows)[:, window_index]
    centers = window_centers(windows)[window_index]
    for row in range(window_index.size):
        members = np.flatnonzero(member[:, row])
        if members.size == 0:
            start, end = windows[window_index[row]]
            logger.warning(
                f"Shot {shot}: window [{start:.3f}, {end:.3f}] s has no grid time "
                "left in the unprocessed file, its profile is left out"
            )
            continue
        fresh[members[np.argmin(np.abs(grid[members] - centers[row]))]] = True
    distance = np.where(member, np.abs(grid[:, None] - centers[None, :]), np.inf)
    nearest = distance.argmin(axis=1)
    draws = np.isfinite(distance.min(axis=1)) if forward_fill else fresh
    sample_index[draws] = nearest[draws]
    return sample_index, fresh


def _fit_anchors(settings: DeviceSettings) -> dict[str, FitAnchors]:
    """Build the per-variable fit anchors from a device's settings.

    Args:
        settings: The device's settings.

    Returns:
        FitAnchors keyed by variable name.
    """
    anchors = {}
    for var in FIT_VARIABLES:
        value_rows = getattr(settings, f"{var}_value_anchors")
        grad_rows = getattr(settings, f"{var}_grad_anchors")
        value = np.asarray(value_rows, dtype=float).reshape(-1, 3)
        grad = np.asarray(grad_rows, dtype=float).reshape(-1, 3)
        anchors[var] = FitAnchors(value=value, grad=grad)
    return anchors


def _anchors_json(anchors: dict[str, FitAnchors]) -> str:
    """Serialize fit anchors for comparison and messages.

    Args:
        anchors: FitAnchors keyed by variable name.

    Returns:
        The anchors as a JSON string.
    """
    return json.dumps(
        {
            var: {"value": a.value.tolist(), "grad": a.grad.tolist()}
            for var, a in anchors.items()
        }
    )


def _tile_channel_groups(groups: list | None, n_columns: int) -> list | None:
    """Stretch per-channel plot masks over pooled rows.

    A pooled row holds every channel once per Thomson sample,
    so a (n_ch,) mask is tiled up to the row width.
    Masks already as wide as the row, (n_t, n_ch) masks, and None pass through.

    Args:
        groups: (mask, color, label) triples from fit_plot_channel_groups.
        n_columns: Width of the staged rows.

    Returns:
        The triples with the masks widened where needed.
    """
    if groups is None:
        return None
    tiled = []
    for mask, color, label in groups:
        mask = np.asarray(mask)
        if mask.ndim == 1 and mask.size != n_columns and n_columns % mask.size == 0:
            mask = np.tile(mask, n_columns // mask.size)
        tiled.append((mask, color, label))
    return tiled


def _hold_equilibrium(
    ds_unprocessed: xr.Dataset, grid: np.ndarray, forward_fill: bool
) -> tuple[dict[str, xr.DataArray], np.ndarray]:
    """Place a shot's equilibria on its 1 kHz grid.

    The equilibrium is reconstructed on its own clock, which is slower than
    the grid on some devices (MAST reconstructs every 5 ms, C-Mod every
    millisecond). The grid times a reconstruction landed on are the ones
    with a finite simagx, the rest hold the last one.

    Args:
        ds_unprocessed: The shot's unprocessed dataset, on the grid, with
            its time coordinate already dropped.
        grid: The shot's 1 kHz timebase [s].
        forward_fill: Hold each reconstruction forward until the next one.

    Returns:
        (equilibrium, fresh): the GEQDSK variables on the grid, and the
        mask of grid times carrying a reconstruction of their own.
    """
    names = [name for name in DATASET_EQUILIBRIUM_SIGNALS if name in ds_unprocessed]
    if "simagx" not in ds_unprocessed:
        return {name: ds_unprocessed[name] for name in names}, np.zeros(
            grid.size, dtype=bool
        )

    reconstructed = np.flatnonzero(
        ds_unprocessed["simagx"]
        .squeeze(EPISODE_DIM, drop=True)
        .transpose(TIME_COORD)
        .notnull()
        .values
    )
    reconstruction_index, fresh = _hold_onto_grid(
        grid, grid[reconstructed], forward_fill
    )
    # reconstruction_index counts reconstructions, the dataset is indexed by
    # grid time, so index the grid times the reconstructions landed on
    ds_held = ds_unprocessed[names].isel(
        {TIME_COORD: reconstructed[np.clip(reconstruction_index, 0, None)]}
    )
    ds_held = ds_held.where(xr.DataArray(reconstruction_index >= 0, dims=TIME_COORD))
    return {name: ds_held[name] for name in names}, fresh
