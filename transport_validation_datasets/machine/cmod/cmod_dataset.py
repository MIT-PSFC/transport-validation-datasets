import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_validation_datasets.dispy_utils import passive_log_settings, summary
from transport_validation_datasets.machine.cmod.dispy_methods import (
    CmodEfitMethods,
    CmodGeometryMethods,
    CmodThomsonMethods,
    UniformTimeSetting,
)
from transport_validation_datasets.machine.generic import (
    make_uniform_1kHz_timebase,
    snap_to_grid,
)
from transport_validation_datasets.workflow import DataWorkflow


class CModDataWorkflow(DataWorkflow):
    """C-Mod specific data workflow for creating and processing datasets."""

    min_pulse_length = 0.5
    min_usable_time = 0.2
    valid_filter = {
        "ip": {"min_abs": 100e3},  # Only care about magnitude of ip
        "n_e_line_average": {"min": 1e18, "max": 4e20},
        "energy_mhd": {"min": 3e3},
        "beta_tor_norm": {"min": 0.0, "max": 2.0},
    }
    transient_filter = {
        "power_ohm": 5.0e6,
        "power_radiated": 2.5e6,
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
            ipmax=self.valid_filter["ip"]["min_abs"],
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
            if self.max_num_shots and (unprocessed_shots >= self.max_num_shots):
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

            if self.shot_already_failed(shot):
                logger.info(f"Shot {shot} failed on a previous run. Skipping.")
                continue

            # Get data for this shot using disruption-py.
            # Three datasets created separately due to timebase differences:
            # fast 0D signals (Ip, B0, shaping, density, power, all native 1 kHz),
            # the EFIT dataset (native 1 kHz on C-Mod),
            # and Thomson scattering (native 20 Hz).
            datasets = []
            missing = False
            for name, getter in (
                ("fast", _get_fast_dataset),
                ("efit", _get_efit_dataset),
                ("thomson", _get_thomson_dataset),
            ):
                ds = getter(shot)
                if ds is None:
                    reason = f"Missing retrievable {name} data."
                    logger.warning(
                        f"Shot {shot} is missing retrievable {name} data. "
                        "Skipping unprocessed data file creation."
                    )
                    self.record_failed_shot(shot, reason)
                    missing = True
                    break
                datasets.append(ds)
            if missing:
                continue

            ds_merged = xr.merge(datasets, compat="no_conflicts", join="outer")

            ds_standardized = self.standardize_signal_names(ds_merged)
            if ds_standardized is None:
                logger.warning(
                    f"Shot {shot} is missing critical signals. Skipping unprocessed data file creation."
                )
                self.record_failed_shot(shot, "Missing critical signals.")
                continue

            ds_unprocessed = self.filter_and_plot(ds_standardized)
            if ds_unprocessed is None:
                logger.warning(
                    f"Shot {shot} did not pass filtering. Skipping unprocessed data file creation."
                )
                self.record_failed_shot(shot, "Did not pass filtering.")
                continue

            ds_unprocessed.to_netcdf(unprocessed_ds_path)
            logger.info(f"Created unprocessed data file for shot {shot}.")
            unprocessed_shots += 1

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals to IMAS-like names, keeping freeqdsk names for EFIT signals.

        All signals stay in SI units. Plasma current and toroidal field are stored
        as magnitudes, sign conventions live in the geqdsk signals.

        Args:
            ds: Merged dataset with disruption-py signal names.

        Returns:
            Dataset with standardized signal names, or None if the shot is missing
            critical signals.
        """
        # disruption-py signal name -> IMAS-like name
        imas_rename = {
            # summary/global_quantities
            "bt": "b0",
            "wmhd": "energy_mhd",
            "beta_n": "beta_tor_norm",
            "p_oh": "power_ohm",
            "p_rad": "power_radiated",
            # summary/heating_current_drive
            "p_icrf": "power_ic",
            "p_lh": "power_lh",
            # summary/line_average
            "n_e": "n_e_line_average",
            # equilibrium/time_slice/boundary
            "a_minor": "minor_radius",
            "kappa": "elongation",
            "tritop": "triangularity_upper",
            "tribot": "triangularity_lower",
            "rout": "geometric_axis_r",
            # thomson_scattering/channel
            "ts_channel_ne": "ts_channel_n_e",
            "ts_channel_ne_error": "ts_channel_n_e_error",
            "ts_channel_te": "ts_channel_t_e",
            "ts_channel_te_error": "ts_channel_t_e_error",
        }

        ds = ds.rename({k: v for k, v in imas_rename.items() if k in ds})

        # TS profiles are the point of the dataset. When get_thomson_channels fails,
        # disruption-py still fills its declared columns, but with NaN on the 0D
        # timebase, so there is no ts_channel dim to plot or fit
        if "ts_channel_n_e" not in ds or "ts_channel" not in ds["ts_channel_n_e"].dims:
            logger.warning("No Thomson scattering channels retrieved for this shot.")
            return None

        # C-Mod has no NBI, zero where ip is valid
        if "ip" in ds:
            ds["power_nbi"] = ds["ip"] * 0.0
            ds["power_nbi"].attrs = {
                "description": "Neutral beam heating power (none on C-Mod)",
                "units": "W",
                "ref": "/summary/heating_current_drive/power_nbi",
            }

        return ds


def _is_empty_result(result: xr.Dataset) -> bool:
    """Check whether get_shots_data returned no usable data for a shot.

    When retrieval fails (e.g. a missing MDSplus tree), get_shots_data logs the
    error and returns an empty dataset with no shot/time index variables. Reshaping
    that with set_index would raise, so callers use this to skip the shot instead.

    Returns:
        True if the result has no usable shot/time data, False otherwise.
    """
    return "shot" not in result or "time" not in result or result["time"].size == 0


def _get_fast_dataset(shot: int) -> xr.Dataset | None:
    """Retrieve fast 0D signals and EFIT dataset.

    Returns:
        Dataset with EFIT signals for the given shot, or None if retrieval
        returned no data.
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
    if _is_empty_result(result):
        return None
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _get_efit_dataset(shot: int) -> xr.Dataset | None:
    """Retrieve EFIT dataset for the given shot.

    Args:
        shot: Shot number to retrieve data for.

    Returns:
        Dataset with GEQDSK signals for the given shot, or None if retrieval
        returned no data.
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
    if _is_empty_result(result):
        return None
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def _get_thomson_dataset(shot: int) -> xr.Dataset | None:
    """Retrieve Thomson scattering data for the given shot.

    Args:
        shot: Shot number to retrieve data for.

    Returns:
        Dataset with Thomson channel signals snapped to the uniform 1 kHz grid,
        or None if retrieval returned no data.
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
    if _is_empty_result(result):
        return None
    # Snap native ~20 Hz TS slices onto the uniform 1 kHz grid, no interpolation.
    # Grid times with no TS slice come back as NaN.
    timebase = make_uniform_1kHz_timebase(float(result["time"].values.max()))
    result = snap_to_grid(result, timebase)
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result
