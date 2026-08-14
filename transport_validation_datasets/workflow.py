from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger

from transport_validation_datasets.gp_fitting.dispatcher import ClusterFitConfig


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
