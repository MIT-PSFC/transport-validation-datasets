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
from transport_validation_datasets.filters import (
    ENERGY_SANITY_LEEWAY,
    FAILURE_MARGIN,
    TRANSIENT_SMOOTHING_WINDOW,
    clip_powers,
    energy_sanity_reason,
    mask_spans,
    radiated_fraction_reason,
    slice_filter_mask,
)
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
    EQUILIBRIUM_HOLD_FLOOR,
    MAX_HOLD_PERIODS,
    POWER_SMOOTHING_WINDOW,
    SOL_EXTENSIONS,
    hold_onto_grid,
    keep_longest_segment,
    kept_segments,
    kept_span,
    reconstruction_clock_period,
    standardize_signal_attrs,
    usable_reconstructions,
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
from transport_validation_datasets.store_schema import (
    DATASET_0D_SIGNALS,
    apply_signal_attrs,
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

# A slice whose fitted te at the LCFS is above this fraction of its peak is unusable, see usable_slice_mask.
# In it11 no C-Mod slice goes above 0.25 and 99 percent of MAST slices stay below 0.15.
FLAT_TE_EDGE_RATIO = 0.4


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
    def min_filter(self) -> dict[str, float]:
        """Minimum thresholds, a grid time where a listed signal is below its threshold is cut out as a gap.

        ip is compared as its magnitude, and its threshold also sets the end of the shot (end_of_shot_index).
        The signals are those of _filter_inputs.

        Returns:
            Thresholds for signals, e.g. {"ip": 1e5}.
        """

    @property
    @abstractmethod
    def max_filter(self) -> dict[str, float]:
        """Maximum thresholds on the raw samples, a grid time where a listed signal is above its threshold is cut out as a gap.

        For records that are broken or extreme, unlike the smoothed transient_filter.
        The signals are those of _filter_inputs, greenwald_fraction included.

        Returns:
            Thresholds for signals, e.g. {"greenwald_fraction": 2.0}.
        """

    @property
    @abstractmethod
    def transient_filter(self) -> dict[str, float]:
        """Dictionary of thresholds for signals, used to filter out transient events.

        Thresholds are compared against the signal smoothed by a centered boxcar
        TRANSIENT_SMOOTHING_WINDOW wide, NOT the raw signal.
        The grid times above a threshold are cut out as a gap, see filter_and_plot.

        Returns:
            Thresholds for signals, e.g. {"signal_name": 1.0}.
        """

    @property
    @abstractmethod
    def end_margin(self) -> float:
        """Margin cut before the end of the plasma, see end_of_shot_index.

        Returns:
            Margin in seconds.
        """

    @property
    @abstractmethod
    def min_pulse_length(self) -> float:
        """Minimum length of the one contiguous segment the filters keep of a shot.

        Returns:
            Minimum pulse length in seconds.
        """

    @property
    @abstractmethod
    def shot_blacklist(self) -> list[int]:
        """List of shots to exclude from processing due to known issues.

        Returns:
            List of shot numbers to exclude.
        """

    # Floor on a shot's mean power_radiated over its kept times, as a fraction of its mean heating power, 0 for none.
    # A dead bolometer reading ~0 W passes the valid filter but breaks every power balance.
    min_radiated_fraction = 0.0
    # Ceiling on the same fraction, inf for none. Above 1 more is radiated than put in.
    max_radiated_fraction = np.inf

    # Shots numbered below this are left out like blacklisted ones, 0 for none.
    first_shot = 0

    # Thomson n_e against the interferometer. Large disagreement indicates TS miscalibration.
    # None to disable this check.
    density_ratio_bounds: tuple[float, float] | None = None

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
        # The standardized source pulls before any filtering, so a filter change reruns without the source
        self.source_data_dir = self.unprocessed_data_dir / "source"
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

    def unprocessed_filter_settings(self) -> dict:
        """Every threshold the unprocessed stage applies (filter_and_plot, shot_rejection_reason).

        A filter rejection's failure note is stamped with these (filter_rejection_note).

        Returns:
            The min, max and transient filters, the transient smoothing window, the failure and end margins,
            the minimum pulse length, and the thresholds of the whole-shot checks.
        """
        return {
            "min_filter": self.min_filter,
            "max_filter": self.max_filter,
            "transient_filter": self.transient_filter,
            "transient_smoothing_window": TRANSIENT_SMOOTHING_WINDOW,
            "failure_margin": FAILURE_MARGIN,
            "end_margin": self.end_margin,
            "min_pulse_length": self.min_pulse_length,
            "min_radiated_fraction": self.min_radiated_fraction,
            "max_radiated_fraction": self.max_radiated_fraction,
            "energy_sanity_leeway": ENERGY_SANITY_LEEWAY,
        }

    def filter_rejection_note(self) -> str:
        """The failure note of a shot the current filters reject.

        It carries the filter settings (unprocessed_filter_settings),
        so a later run can tell whether the same filters would judge the shot again.

        Returns:
            The note, JSON with sorted keys, so the same settings always give the same text.
        """
        note = {
            "reason": "Did not pass filtering.",
            "filters": self.unprocessed_filter_settings(),
        }
        return to_json(note)

    def shot_rejected_by_current_filters(self, shot: int) -> bool:
        """Check whether the current filter settings rejected a shot on a previous run.

        Args:
            shot: Shot number to check.

        Returns:
            True if the shot's failure note is the one a rejection would write now (filter_rejection_note).
        """
        note_path = self.failed_shots_dir / f"{shot}.txt"
        if not note_path.exists():
            return False
        note_recorded = note_path.read_text()
        note_current = self.filter_rejection_note()
        return note_recorded == note_current

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

        Every source pull is also kept unfiltered in source_data_dir.
        Resumes: shots that already have a file or are excluded (excluded_shot_reason) are skipped.
        A shot with a kept source pull is filtered from it without touching the source.
        A shot the current filter settings rejected on a previous run is skipped (shot_rejected_by_current_filters),
        and a change to the settings filters it again.
        A change to the filter code alone does not, so it needs failed_shots_dir cleared.
        Deleting the unprocessed files reruns a filter change on the shots that passed.
        A shot that failed on an earlier run and has no kept pull is skipped.
        Source reads run prepare_workers at a time, while filtering, plotting, and writing stay on
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
                exclusion_reason = self.excluded_shot_reason(shot)
                if exclusion_reason is not None:
                    logger.info(f"Shot {shot} is {exclusion_reason}. Skipping.")
                    continue
                if (self.unprocessed_data_dir / f"{shot}.nc").exists():
                    logger.info(
                        f"Unprocessed data file for shot {shot} already exists. Skipping."
                    )
                    n_files += 1
                    continue
                has_source_pull = (self.source_data_dir / f"{shot}.nc").exists()
                if has_source_pull and self.shot_rejected_by_current_filters(shot):
                    logger.info(
                        f"Shot {shot} was rejected by the current filters on a previous run. Skipping."
                    )
                    continue
                if not has_source_pull and self.shot_already_failed(shot):
                    logger.info(f"Shot {shot} failed on a previous run. Skipping.")
                    continue
                batch.append(shot)
            n_files += self._read_and_write_shots(batch, workers)

        logger.info(f"Finished with {n_files} {self.ds_name} unprocessed data files.")

    def _read_and_write_shots(self, shots: list[int], workers: int) -> int:
        """Read a batch of shots, then filter and write them.

        A shot with a pull in source_data_dir is loaded from it,
        the rest are read from the source and their pulls kept there.

        Args:
            shots: Shot numbers to read.
            workers: Threads to read them with.

        Returns:
            How many unprocessed data files were written.
        """
        if not shots:
            return 0
        source_pull_paths = {
            shot: self.source_data_dir / f"{shot}.nc" for shot in shots
        }
        shots_to_read = [shot for shot in shots if not source_pull_paths[shot].exists()]

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

        if len(shots_to_read) <= 1 or workers == 1:
            datasets_read = [read(shot) for shot in shots_to_read]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                datasets_read = list(pool.map(read, shots_to_read))
        datasets_by_shot = dict(zip(shots_to_read, datasets_read, strict=True))

        n_written = 0
        for shot in shots:
            if shot in datasets_by_shot:
                ds_standardized = datasets_by_shot[shot]
                if ds_standardized is None:
                    continue
                ds_standardized.attrs = source_provenance(ds_standardized.attrs)
                self.source_data_dir.mkdir(parents=True, exist_ok=True)
                ds_standardized.to_netcdf(source_pull_paths[shot])
            else:
                ds_standardized = xr.load_dataset(source_pull_paths[shot])
                logger.info(f"Shot {shot}: filtering the kept source pull")
            ds_unprocessed = self.filter_and_plot(ds_standardized)
            if ds_unprocessed is None:
                logger.warning(
                    f"Shot {shot} did not pass filtering. Skipping unprocessed data file creation."
                )
                self.record_failed_shot(shot, self.filter_rejection_note())
                continue
            ds_unprocessed = clip_powers(ds_unprocessed)
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

        Every check cuts the grid times it fails out as a gap:
        the end of the shot (end_of_shot_index),
        a 0D signal of DATASET_0D_SIGNALS that is not finite,
        a min_filter signal below its threshold or a max_filter signal above it,
        and a transient_filter signal above its threshold after smoothing.
        What survives is one contiguous segment.
        Each segment is started where a kept usable reconstruction (usable_reconstructions) reaches,
        and only the longest segment is kept.
        The shot is rejected when ip never reaches its min_filter threshold,
        when that segment is shorter than min_pulse_length,
        or when shot_rejection_reason finds a broken record in it.

        Rejected shots are plotted unfiltered to rejected_shots_dir.
        Accepted shots are plotted unfiltered to accepted_shots_dir,
        with the kept segment shaded green.
        Both plots mark the end margin cutoff and shade the transients red.

        Args:
            ds_input: Dataset with standardized signal names for one shot.

        Returns:
            The filtered dataset, or None if the shot should be rejected.
        """
        shot = ds_input.shot.values[0]
        times = ds_input["time"].values
        ds_shot = ds_input.squeeze(EPISODE_DIM, drop=True)

        # 0 to 2: The end of the shot, the 0D checks and the transients, see slice_filter_mask
        slice_filter = slice_filter_mask(
            ds_shot,
            times,
            self.min_filter,
            self.max_filter,
            self.transient_filter,
            self.end_margin,
        )
        if slice_filter is None:
            logger.warning(
                f"Shot {shot} rejected: |ip| never reaches {self.min_filter['ip']:g} A"
            )
            self._plot_unprocessed(ds_input, shot, None, None, None)
            return None
        filtered_mask, transient_time_mask, end_cut_index = slice_filter
        end_margin_time = float(times[np.clip(end_cut_index, 0, times.size - 1)])
        n_transient = int(transient_time_mask.sum())
        if n_transient:
            logger.debug(f"Shot {shot}: transients at {n_transient} grid times")
        transient_spans = mask_spans(transient_time_mask, times)

        # 3: Start each segment where the store will have an equilibrium, then keep only the longest.
        # The store only holds the reconstructions that are kept (_hold_equilibrium),
        # so a segment's first grid times have none when the one before it is cut.
        if "simagx" in ds_input:
            reconstruction_usable = usable_reconstructions(ds_input)
            clock_period = reconstruction_clock_period(ds_input, times)
            kept_mask, dropped_lengths, n_trimmed = _trim_and_keep_longest(
                filtered_mask,
                times,
                reconstruction_usable,
                clock_period,
            )
            if n_trimmed:
                logger.debug(
                    f"Shot {shot}: trimmed {n_trimmed} grid times with no equilibrium in reach "
                    f"from the starts of their segments"
                )
        else:
            kept_mask, dropped_lengths = keep_longest_segment(filtered_mask, times)
        if dropped_lengths:
            logger.info(
                f"Shot {shot}: kept the longest segment, dropped {len(dropped_lengths)} shorter one(s), "
                f"the longest {1e3 * max(dropped_lengths):.0f} ms, {1e3 * sum(dropped_lengths):.0f} ms in all"
            )
        valid_mask = xr.DataArray(
            kept_mask[np.newaxis, :],
            coords={EPISODE_DIM: ds_input[EPISODE_DIM].values, "time": times},
            dims=(EPISODE_DIM, "time"),
        )

        # Load-bearing broadcast: valid_mask carries the shot and time dims, so
        # this also gives every static quantity (the limiter contour, the fixed
        # grid extents, C-Mod's fixed channel positions) a time axis.
        # The internal dataset carries them per slice like everything else, and
        # _hold_equilibrium indexes them by grid time.
        ds_filtered = ds_input.where(valid_mask, drop=True)

        # 4: Reject the shot when the kept segment is shorter than min_pulse_length
        pulse_length = kept_span(kept_mask, times)
        if pulse_length < self.min_pulse_length:
            rejection_reason = (
                f"longest segment {pulse_length:.3f} s after filtering, "
                f"shorter than min_pulse_length {self.min_pulse_length} s"
            )
        else:
            # 5: Whole-shot checks on what survived, see shot_rejection_reason
            rejection_reason = self.shot_rejection_reason(ds_filtered)

        if rejection_reason is not None:
            logger.warning(f"Shot {shot} rejected: {rejection_reason}")
            self._plot_unprocessed(
                ds_input, shot, end_margin_time, transient_spans, None
            )
            return None
        # Plot the entire shot, with the kept segment shaded green
        kept_spans = mask_spans(kept_mask, times)
        self._plot_unprocessed(
            ds_input, shot, end_margin_time, transient_spans, kept_spans
        )
        return ds_filtered

    def _plot_unprocessed(
        self,
        ds_input: xr.Dataset,
        shot: int,
        end_margin_time: float | None,
        transient_spans: list[tuple[float, float]] | None,
        kept_spans: list[tuple[float, float]] | None,
    ):
        """Plot one shot's unfiltered signals with the filter thresholds, see plot_unprocessed_data.

        A shot without kept_spans was rejected and goes to rejected_shots_dir,
        one with them to accepted_shots_dir.

        Args:
            ds_input: Dataset with standardized signal names for one shot.
            shot: Shot number.
            end_margin_time: Time of the end-of-shot cut [s], None when there is none.
            transient_spans: (start, end) intervals cut out as transients.
            kept_spans: (start, end) intervals kept, None for a rejected shot.
        """
        if kept_spans is None:
            fig_path = self.rejected_shots_dir / f"{shot}.png"
            title = f"Shot {shot} (REJECTED)"
            window_spans = None
        else:
            fig_path = self.accepted_shots_dir / f"{shot}.png"
            title = f"Shot {shot}"
            window_spans = (
                None if self.shot_windows is None else self.shot_windows.get(int(shot))
            )
        plot_unprocessed_data(
            ds_input,
            fig_path,
            title=title,
            min_filter=self.min_filter,
            max_filter=self.max_filter,
            transient_filter=self.transient_filter,
            end_margin_time=end_margin_time,
            transient_spans=transient_spans,
            kept_spans=kept_spans,
            window_spans=window_spans,
        )

    def shot_rejection_reason(self, ds: xr.Dataset) -> str | None:
        """Check one shot's kept times for broken records the filters let through.

        Runs on the filtered dataset in filter_and_plot,
        and again on the unprocessed file at the stack stage,
        so a file written before a check existed is judged by the code as it is now.
        The checks, in order, are the shared ones of filters.py:
        a dead bolometer or more radiated than put in (radiated_fraction_reason, against min_ and max_radiated_fraction),
        then a stored-energy rise the input power cannot explain (energy_sanity_reason).
        Both clip the powers themselves, so the clipped unprocessed file reaches the same verdict.

        Args:
            ds: One shot's dataset on its kept times, with standardized names.

        Returns:
            Why the shot is rejected, or None if it passes.
        """
        ds_shot = ds.squeeze(EPISODE_DIM, drop=True)
        radiated_reason = radiated_fraction_reason(
            ds_shot, self.min_radiated_fraction, self.max_radiated_fraction
        )
        if radiated_reason is not None:
            return radiated_reason
        return energy_sanity_reason(ds_shot)

    def excluded_shot_reason(self, shot: int) -> str | None:
        """Check whether a shot is left out by number, before any of its data is looked at.

        Applied before the source read and again at the stack stage,
        so a shot excluded after its files were written stays out of the store.

        Args:
            shot: Shot number.

        Returns:
            Why the shot is left out, or None if it is not.
        """
        if shot in self.shot_blacklist:
            return "blacklisted"
        if shot < self.first_shot:
            return f"before first_shot {self.first_shot}"
        return None

    def fit_rejection_reason(
        self, ds_fit: xr.Dataset, ds_unprocessed: xr.Dataset
    ) -> str | None:
        """Check a shot's fitted density against its line-averaged density.

        Thomson and the interferometer measure the same density,
        so a shot whose fits sit far off the interferometer has a broken Thomson calibration.
        Each slice's ratio is the mean of the fitted n_e over rho_tor_norm in [0, 1]
        over n_e_line_average at the slice time.
        The shot median of the ratios must lie inside density_ratio_bounds.
        The mean over rho stands in for the chord integral, whose geometry neither device records.
        Its offset from 1 depends on the chord and the profile shapes, which the per-device bounds absorb.

        Args:
            ds_fit: One shot's usable fitted slices, with their time coordinate.
            ds_unprocessed: The shot's unprocessed dataset.

        Returns:
            Why the shot is rejected, or None if it passes or the device has no gate.
        """
        if (
            self.density_ratio_bounds is None
            or "n_e_line_average" not in ds_unprocessed
        ):
            return None
        grid = np.asarray(ds_unprocessed[TIME_COORD].values, dtype=float)
        line_average = np.asarray(
            ds_unprocessed["n_e_line_average"].squeeze(EPISODE_DIM, drop=True).values,
            dtype=float,
        )
        has_line_average = np.isfinite(line_average)
        if not has_line_average.any():
            return "no finite n_e_line_average to compare the fitted n_e with"
        slice_times = np.asarray(
            ds_fit[TIME_COORD].squeeze(EPISODE_DIM, drop=True).values, dtype=float
        )
        line_average_at_slices = np.interp(
            slice_times, grid[has_line_average], line_average[has_line_average]
        )
        inside_lcfs = ds_fit["rho_tor_norm"].values <= 1.0
        ne_fit = (
            ds_fit["n_e"]
            .squeeze(EPISODE_DIM, drop=True)
            .transpose(TIME_DIM, "rho_tor_norm")
            .values
        )
        ne_fit_mean = ne_fit[:, inside_lcfs].mean(axis=1)
        ratio = ne_fit_mean / line_average_at_slices
        # Only the unfit rows a store kept on request are NaN, and they say nothing
        ratio_finite = ratio[np.isfinite(ratio)]
        if ratio_finite.size == 0:
            return "no fitted slice with an n_e_line_average to compare with"
        ratio_median = float(np.median(ratio_finite))
        ratio_low, ratio_high = self.density_ratio_bounds
        if not ratio_low <= ratio_median <= ratio_high:
            return (
                f"fitted n_e averages {ratio_median:.2f}x n_e_line_average, "
                f"outside {ratio_low}-{ratio_high}"
            )
        return None

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

    def run_gp_fitting(self, max_pages: int | None = None, skip_plots: bool = False):
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
        4. Plot the fits per shot into fit_plots_dir, unless skip_plots.
           The plots take about a minute per shot, far longer than the rest on a cluster.
           A later run without skip_plots plots the shots that have no PDF yet.

        Args:
            max_pages: Maximum number of pages to plot per shot. None plots all.
            skip_plots: Leave out step 4.
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
        if not skip_plots:
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

    def fit_plot_channel_groups(
        self, shot: int, fit_input: ShotFitInput
    ) -> list | None:
        """Get the channel grouping used to color the fit plots.

        Subclasses can split channels by diagnostic (C-Mod core vs edge Thomson)
        or by position (MAST inboard vs outboard branch, which moves slice to slice).
        The base implementation plots them as one group.

        Args:
            shot: Shot number being plotted.
            fit_input: The shot's staged fit input, whose rows the masks must match.

        Returns:
            (mask, color, label) triples, each mask (n_ch,) or (n_rows, n_columns), or None for a single group.
        """
        return None

    def fit_plot_dropped_readings(self, shot: int) -> tuple | None:
        """Get the readings a device drops from every fit, to mark on the fit plots.

        A device that drops a faulty channel from every shot returns its readings here,
        since the staged batch no longer holds them.
        The base implementation drops none.

        Args:
            shot: Shot number being plotted.

        Returns:
            (times, rho_tor_norm, {var: (y, err)}) with the (n_samples,) Thomson sample times [s]
            and (n_samples, n_dropped) arrays in the fit units, a variable absent when it has none,
            or None.
        """
        return None

    def plot_fit_results(self, max_pages: int | None = None):
        """Plot the GP fits of every fitted shot, one PDF per shot.

        Plots the exact (cleaned, floored) channel data the fit consumed,
        straight from the staged batch files,
        and in red the readings the device drops from every fit (fit_plot_dropped_readings).
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
                dropped_readings = self.fit_plot_dropped_readings(shot)
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
                        self.fit_plot_channel_groups(shot, si), si.x.shape[1]
                    ),
                    max_pages=max_pages,
                    window_bounds=window_bounds,
                    dropped_readings=dropped_readings,
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
            mb_per_chunk: Target size of each variable's storage chunks,
                chunked along EPISODE_DIM only. None leaves the chunking alone.
            drop_unfit_slices: Ignore the slices usable_slice_mask rejects,
                as though the shot had no Thomson sample there.
                False places every slice on the grid, all-NaN profiles included.
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
            result file, less the excluded ones (excluded_shot_reason).
        """
        unprocessed = {int(p.stem) for p in self.unprocessed_data_dir.glob("*.nc")}
        fitted = {int(p.stem) for p in self.fit_shots_dir.glob("*.nc")}
        missing = unprocessed - fitted
        if missing:
            logger.warning(
                f"{len(missing)} unprocessed shots have no '{self.fit_method}' fit results yet "
                f"and are left out of the internal dataset"
            )
        excluded = {
            shot
            for shot in unprocessed & fitted
            if self.excluded_shot_reason(shot) is not None
        }
        if excluded:
            logger.info(
                f"{len(excluded)} shots are blacklisted or before first_shot "
                f"and are left out of the internal dataset"
            )
        return sorted((unprocessed & fitted) - excluded)

    def export_to_imas(self, overwrite: bool = False):
        """Writes every fitted shot's equilibrium/core_profiles/summary/wall to IMAS format.

        Exports the shots the stores take, through the same checks (_usable_shot_fit).

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

        shots = self._internal_dataset_shots()
        if not shots:
            logger.warning(
                "No shots have both unprocessed data and fit results; nothing to export to IMAS"
            )
            return

        n_written = 0
        n_rejected = 0
        for shot in shots:
            shot_dir = self.imas_export_dir / str(shot)
            if (shot_dir / "core_profiles.nc").exists() and not overwrite:
                continue
            with xr.open_dataset(self.fit_shots_dir / f"{shot}.nc") as ds_file:
                fit_ds = ds_file.load()
            with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds_file:
                unprocessed_ds = ds_file.load()
            # The shots the stores leave out stay out of the export too
            fit_ds = self._usable_shot_fit(shot, unprocessed_ds, fit_ds, True)
            if fit_ds is None:
                n_rejected += 1
                continue
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
            f"Wrote IMAS output for {n_written} of {len(shots)} shots to {self.imas_export_dir}, "
            f"{n_rejected} rejected by the checks of the stores, "
            f"{len(shots) - n_written - n_rejected} already up to date or failed"
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

    def _usable_shot_fit(
        self,
        shot: int,
        ds_unprocessed: xr.Dataset,
        ds_fit: xr.Dataset,
        drop_unfit_slices: bool,
    ) -> xr.Dataset | None:
        """Run the whole-shot checks of the stores on a shot's files and keep its usable slices.

        The stack stage and export_to_imas both go through here, so they leave out the same shots.
        The checks, in order:
        shot_rejection_reason on the unprocessed file,
        at least one slice left (usable_slice_mask),
        and fit_rejection_reason on the slices left.
        A rejected shot is logged with its reason.

        Args:
            shot: Shot number, for the log.
            ds_unprocessed: The shot's unprocessed dataset.
            ds_fit: The shot's fit result dataset.
            drop_unfit_slices: Keep only the slices usable_slice_mask accepts. False keeps every slice.

        Returns:
            The fit dataset cut to the kept slices, or None when a check rejects the shot.
        """
        rejection_reason = self.shot_rejection_reason(ds_unprocessed)
        if rejection_reason is None:
            keep = (
                usable_slice_mask(ds_fit)
                if drop_unfit_slices
                else np.ones(ds_fit.sizes[TIME_DIM], dtype=bool)
            )
            ds_fit = ds_fit.isel({TIME_DIM: np.flatnonzero(keep)})
            if keep.any():
                rejection_reason = self.fit_rejection_reason(ds_fit, ds_unprocessed)
            else:
                rejection_reason = f"no usable {self.fit_method} fits"
        if rejection_reason is not None:
            logger.warning(f"Shot {shot}: {rejection_reason}, leaving it out")
            return None
        return ds_fit

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
            drop_unfit_slices: Ignore the slices usable_slice_mask rejects,
                as though the shot had no Thomson sample there.
            forward_fill: Hold profiles and equilibria forward onto the grid
                times between their samples.

        Returns:
            The shot's dataset, or None when _usable_shot_fit rejects it.

        Raises:
            ValueError: If the fitted slice times are not on the unprocessed
                file's timebase, or no time window overlaps it.
        """
        with xr.open_dataset(self.unprocessed_data_dir / f"{shot}.nc") as ds_file:
            ds_unprocessed = ds_file.load()
        with xr.open_dataset(self.fit_shots_dir / f"{shot}.nc") as ds_file:
            ds_fit = ds_file.load()
        fit_mode = ds_fit.attrs["fit_mode"]
        # The full window list, before the unusable rows go:
        # a window whose every fit was culled still bounds the stored grid
        windows = _fit_windows(ds_fit)
        ds_fit = self._usable_shot_fit(shot, ds_unprocessed, ds_fit, drop_unfit_slices)
        if ds_fit is None:
            return None
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
            slice_index, fresh_profile = hold_onto_grid(
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
        # The radius b0 is given at, per shot, from the unprocessed file's attribute
        r0 = np.array([ds_unprocessed.attrs["r0"]], dtype=np.float32)
        ds_stacked["r0"] = xr.DataArray(r0, dims=(EPISODE_DIM,))
        apply_signal_attrs(ds_stacked, self.signal_attrs)
        standardize_signal_attrs(ds_stacked)

        # The sign convention of the GEQDSK block, per shot since a dataset
        # may mix field directions, from the unprocessed file's attribute (see cocos_from_signs)
        cocos = ds_unprocessed.attrs.get("cocos")
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
        # cocos and r0 are per shot, the store carries them as variables
        attrs.update(
            merge_shot_attrs(per_shot, exclude=(*SOURCE_VOLATILE_KEYS, "cocos", "r0"))
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
            threshold the unprocessed and stack stages applied), and fit_settings (the
            staging knobs the fits were made with,
            including the full fit grid the store's rho_tor_norm coordinate is cut from).
        """
        unprocessed_filters = self.unprocessed_filter_settings()
        filters = {
            **unprocessed_filters,
            "shot_blacklist": list(self.shot_blacklist),
            "first_shot": self.first_shot,
            "density_ratio_bounds": self.density_ratio_bounds,
            "flat_te_edge_ratio": FLAT_TE_EDGE_RATIO,
        }
        return {
            "device_settings": to_json(self.settings),
            "filters": to_json(filters),
            "fit_settings": to_json(
                {
                    "rho_tor_norm_grid": self.fit_rho_tor_norm,
                    "min_points": self.fit_min_points,
                    "scale_per_slice": self.fit_scale_per_slice,
                    "bounds": self.fit_bounds,
                    "max_hold_periods": MAX_HOLD_PERIODS,
                    "equilibrium_hold_floor": EQUILIBRIUM_HOLD_FLOOR,
                    "power_smoothing_window": POWER_SMOOTHING_WINDOW,
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


def _trim_segment_starts(keep: np.ndarray, can_start: np.ndarray) -> np.ndarray:
    """Clear the samples of each run of kept samples ahead of its first sample that can start one.

    A run is a stretch of consecutive kept samples on the grid, as in keep_longest_segment.
    Each run is cut from the front only:
    its samples before the first one marked in can_start are cleared,
    and that sample and everything after it stay, whatever can_start says further on.
    A run with no sample marked in can_start is cleared whole.
    can_start outside the runs is ignored, a marked sample just before a run does not start it.

    _trim_and_keep_longest passes the grid times that a usable reconstruction of the same run reaches within the hold,
    so every kept segment starts where the store will have an equilibrium.
    Equilibrium lapses inside a segment are left for the stack stage to show as gaps.
    It runs before keep_longest_segment, so a trimmed run is judged on what is left of it.

    Example, with 1 for True:
        keep      1 1 1 0 1 1 1 1 0 1 1
        can_start 1 0 0 0 0 0 1 0 1 0 0
        returned  1 1 1 0 0 0 1 1 0 0 0

    Args:
        keep: Mask over the uniform 1 kHz grid, True where the sample survived the filters. Not modified.
        can_start: Mask over the same grid, True where a run may start.

    Returns:
        A copy of keep with every run starting on a sample of can_start.
    """
    keep = np.asarray(keep, dtype=bool).copy()
    starts, ends = kept_segments(keep)
    for start, end in zip(starts, ends):
        startable = np.flatnonzero(can_start[start:end])
        first_startable = startable[0] if startable.size else end - start
        keep[start : start + first_startable] = False
    return keep


def _trim_and_keep_longest(
    keep: np.ndarray,
    times: np.ndarray,
    reconstruction_usable: np.ndarray,
    clock_period: float,
) -> tuple[np.ndarray, list[float], int]:
    """Start each kept segment where one of its own reconstructions reaches, and keep only the longest.

    A grid time has an equilibrium in the store when a kept usable reconstruction
    at or before it is within the hold of _hold_equilibrium
    (MAX_HOLD_PERIODS of the clock period, at least EQUILIBRIUM_HOLD_FLOOR).
    Only the segment kept here reaches the store, so only its own reconstructions are kept,
    and a reconstruction in an earlier segment never starts a later one.
    _trim_segment_starts cuts each segment's leading grid times that none of its own reconstructions reaches,
    then keep_longest_segment judges the segments on what is left.
    Since each segment is trimmed as it would be if it alone were kept, one pass is exact.

    Args:
        keep: Mask over the uniform 1 kHz grid, True where the sample survived the filters. Not modified.
        times: The grid times [s].
        reconstruction_usable: Mask over the grid, True at the usable reconstructions (usable_reconstructions).
        clock_period: The reconstruction clock's period [s] (reconstruction_clock_period).

    Returns:
        (keep, dropped_lengths, n_trimmed): the new mask,
        the lengths [s] of the segments dropped for a longer one,
        and how many grid times the trims cut.
    """
    keep = np.asarray(keep, dtype=bool)
    reconstruction_kept = np.flatnonzero(reconstruction_usable & keep)
    held_index, _ = hold_onto_grid(
        times,
        times[reconstruction_kept],
        True,
        clock_period,
        hold_floor=EQUILIBRIUM_HOLD_FLOOR,
    )
    has_held = held_index >= 0
    # Grid index of the reconstruction each grid time holds, and of the start of the run it sits in
    held_row = np.full(keep.size, -1)
    held_row[has_held] = reconstruction_kept[held_index[has_held]]
    starts, _ = kept_segments(keep)
    is_start = np.zeros(keep.size, dtype=bool)
    is_start[starts] = True
    index_if_start = np.where(is_start, np.arange(keep.size), -1)
    run_start = np.maximum.accumulate(index_if_start)
    can_start = has_held & (held_row >= run_start)
    keep_trimmed = _trim_segment_starts(keep, can_start)
    n_trimmed = int(keep.sum() - keep_trimmed.sum())
    keep_longest, dropped_lengths = keep_longest_segment(keep_trimmed, times)
    return keep_longest, dropped_lengths, n_trimmed


def usable_slice_mask(ds_fit: xr.Dataset) -> np.ndarray:
    """Find the slices of a shot's fit that hold a usable te and ne profile.

    A slice is unusable when either fit did not come back usable (USABLE_FIT_STATUSES),
    or when a profile that did come back fails a screen:
    - flat te: te at the LCFS above FLAT_TE_EDGE_RATIO of its peak.
      MAST slices whose outboard Thomson reads far hotter than the inboard fit this flat,
      most likely an equilibrium that maps the two branches to the wrong flux surfaces.
    - te band, ne band: the 1 sigma band inside the LCFS wider than the profile's peak,
      one channel's huge error carried straight into the band.
    The screens are method agnostic and run at the stack stage, so retuning them needs a restack, not a refit.
    Logs how many slices each check rejects.

    Args:
        ds_fit: One shot's fit result dataset.

    Returns:
        (n_t,) boolean mask over the shot's slices.
    """
    ds_shot = ds_fit.squeeze(EPISODE_DIM)
    shot = int(ds_shot[EPISODE_DIM])
    unusable = {}
    status_usable = np.ones(ds_shot.sizes[TIME_DIM], dtype=bool)
    for name in ("t_e_fit_status", "n_e_fit_status"):
        status = ds_shot[name].values
        status_usable &= np.isin(status, USABLE_FIT_STATUSES)
    unusable["fit status"] = ~status_usable
    te_edge = ds_shot["t_e"].interp(rho_tor_norm=1.0).values
    te_peak = ds_shot["t_e"].max("rho_tor_norm").values
    unusable["flat te"] = te_edge > FLAT_TE_EDGE_RATIO * te_peak
    # Past the LCFS the band holds the outermost channel's error flat, which says nothing about the profile inside
    inside_lcfs = np.flatnonzero(ds_shot["rho_tor_norm"].values <= 1.0)
    ds_core = ds_shot.isel(rho_tor_norm=inside_lcfs)
    for name in ("t_e", "n_e"):
        error_peak = ds_core[f"{name}_error"].max("rho_tor_norm").values
        profile_peak = ds_core[name].max("rho_tor_norm").values
        unusable[f"{name} band"] = error_peak > profile_peak
    unusable_masks = list(unusable.values())
    unusable_any = np.logical_or.reduce(unusable_masks)
    if unusable_any.any():
        rejected_counts = {
            check: int(mask.sum()) for check, mask in unusable.items() if mask.any()
        }
        logger.debug(
            f"Shot {shot}: {int(unusable_any.sum())} of {unusable_any.size} "
            f"fitted slices unusable, per check {rejected_counts}"
        )
    return ~unusable_any


def _hold_fits_onto_grid(ds_fit: xr.Dataset, slice_index: np.ndarray) -> xr.Dataset:
    """Place a shot's fit results on its 1 kHz grid.

    Args:
        ds_fit: The shot's fit results, one row per Thomson slice.
        slice_index: Slice each grid time draws from, from hold_onto_grid.

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
        (the caller then falls back to hold_onto_grid's own estimate).
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
        (sample_index, fresh) as hold_onto_grid returns them: the fit row
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
    Each is held for up to MAX_HOLD_PERIODS of the clock, or EQUILIBRIUM_HOLD_FLOOR when that is longer,
    so a few missing reconstructions are bridged.
    An unusable reconstruction (usable_reconstructions) is treated as missing,
    so the previous one holds over it and it is not fresh.

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

    clock_period = reconstruction_clock_period(ds_unprocessed, grid)
    usable = usable_reconstructions(ds_unprocessed)
    reconstructed = np.flatnonzero(usable)
    reconstruction_index, fresh = hold_onto_grid(
        grid,
        grid[reconstructed],
        forward_fill,
        clock_period,
        hold_floor=EQUILIBRIUM_HOLD_FLOOR,
    )
    # reconstruction_index counts reconstructions, the dataset is indexed by
    # grid time, so index the grid times the reconstructions landed on
    ds_held = ds_unprocessed[names].isel(
        {TIME_COORD: reconstructed[np.clip(reconstruction_index, 0, None)]}
    )
    ds_held = ds_held.where(xr.DataArray(reconstruction_index >= 0, dims=TIME_COORD))
    return {name: ds_held[name] for name in names}, fresh
