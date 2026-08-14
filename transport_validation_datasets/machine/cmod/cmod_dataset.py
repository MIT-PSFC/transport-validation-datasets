import numpy as np
import xarray as xr
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_validation_datasets.dispy_utils import passive_log_settings, summary
from transport_validation_datasets.workflow import DataWorkflow
from transport_validation_datasets.machine.cmod.dispy_methods import CmodGeometryMethods, CmodEfitMethods, CmodThomsonMethods, UniformTimeSetting


class CModDataWorkflow(DataWorkflow):
    """C-Mod specific data workflow for creating and processing datasets."""

    min_pulse_length = 0.5
    valid_filter = {
        "ip": {"min": 100e3},
        "n_e_line_average": {"min": 1e18, "max": 4e20},
        "energy_mhd": {"min": 3e3, "max": 2e6},
        "beta_tor_norm": {"min": 0.0, "max": 2.0},
    }
    transient_filter = {
        "p_oh": 5.0,
        "p_rad": 2.5,
    }
    end_margin = 0.02
    shot_blacklist = []

    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from C-Mod SQL database.

        Returns:
            Shot numbers to process, filtered to days with blessed Thomson data.
        """
        data = summary(
            summary_table="summary",
            ipmax=self.valid_filter["ip"]["min"],
            pulse_length=self.min_pulse_length,
            min_shot=1160500000,
            max_shot=1160932000,
            shots=False,
        )
        shotlist = data[:, 0].astype(int).tolist()

        # Days with blessed TS data, email from J. Hughes 2025-12-12
        blessed_days = [
            1160503,
            1160527,
            1160621,
            1160628,
            1160630,
            1160708,
        ]
        blessed_day_ranges = [
            [1160607, 1160610],
            [1160712, 1160719],
            [1160803, 1160820],
            [1160823, 1160903],
            [1160908, 1160916],
            [1160919, 1160924],
            [1160927, 1160931],
        ]
        for day_range in blessed_day_ranges:
            blessed_days.extend(range(day_range[0], day_range[1] + 1))

        # Filter to blessed Thomson days
        shotlist = [shot for shot in shotlist if shot // 1000 in blessed_days]
        return shotlist

    def make_unprocessed_data_files(self):
        """Create unprocessed data files for each shot in the shotlist.

        Unprocessed data files contain everything needed to create the final dataset.
        Signals have standardized names and are on a common timebase

        """
        unprocessed_shots = 0
        for shot in self.shotlist:
            if unprocessed_shots >= self.max_num_shots:
                logger.info(
                    f"Reached maximum number of unprocessed shots ({self.max_num_shots}). Stopping."
                )
                break

            if shot in self.shot_blacklist:
                logger.info(f"Shot {shot} is blacklisted. Skipping.")
                continue

            unprocessed_ds_path = self.unprocessed_data_dir / f"{shot}.nc"
            if unprocessed_ds_path.exists():
                logger.info(
                    f"Unprocessed data file for shot {shot} already exists. Skipping."
                )
                unprocessed_shots += 1
                continue

            # Get data for this shot using disruption-py
            # Two datasets created separately due to timebase differences
            # One for fast 0D signals (Ip, B0, shaping, density, power, all native 1 kHz)
            # One for the EFIT dataset (native 1 kHz on C-Mod)
            # And one for Thomson scattering (native 20 Hz)
            ds_fast = _get_fast_dataset(shot)
            ds_efit = _get_efit_dataset(shot)
            ds_thomson = _get_thomson_dataset(shot)

            unprocessed_shots += 1


    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals in dataset to typical names.
        
        Returns:
            Dataset with standardized signal names, or None if critical signals are missing
        """
        return None


def _get_fast_dataset(shot: int) -> xr.Dataset:
    """Retrieve fast 0D signals and EFIT dataset.

    Returns:
        Dataset with EFIT signals for the given shot.
    """
    cmod_dataset_signals = [
        "ip",  # Plasma current
        "bt",  # On-axis magnetic field
        "wmhd",  # Total stored energy (C-Mod has no consistent fast particle measurement, so this is all we've got)
        "n_e",  # Line average electron density [m^-3]
        "beta_n",  # Normalized beta
        "a_minor",  # Plasma minor radius
        "kappa",  # Plasma elongation
        "tritop",  # Top triangularity
        "tribot",  # Bottom triangularity
        "rout",  # Geometric major radius [m]
        # Power sources and sinks
        "p_oh",  # Ohmic heating power
        "p_rad",  # Bulk radiated heating power
        "p_icrf",  # ICRF heating power
        "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    ]

    retrieval_settings = RetrievalSettings(
        run_columns=cmod_dataset_signals,
        time_setting=UniformTimeSetting(),
        efit_nickname_setting="EFIT21",
        only_requested_columns=True,
        custom_physics_methods=[CmodGeometryMethods.get_geometric_major_radius],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shot,
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _get_efit_dataset(shot: int) -> xr.Dataset:
    """Retrieve EFIT dataset for the given shot.

    Args:
        shot: Shot number to retrieve data for.
    """
    settings = RetrievalSettings(
        run_methods=["get_geqdsk_parameters"],
        efit_nickname_setting="EFIT21",
        time_setting=UniformTimeSetting(),
        custom_physics_methods=[CmodEfitMethods.get_geqdsk_parameters],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shot,
        retrieval_settings=settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result

def _get_thomson_dataset(shot: int) -> xr.Dataset:
    """Retrieve Thomson scattering data for the given shot.

    Args:
        shot: Shot number to retrieve data for.
    """

    retrieval_settings = RetrievalSettings(
        run_methods=["get_thomson_channels"],
        efit_nickname_setting="EFIT21",
        only_requested_columns=False,
        custom_physics_methods=[CmodThomsonMethods.get_thomson_channels],
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=[shot],
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result

