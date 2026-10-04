"""DIII-D data workflow, built from MDSplus through disruption-py and the IDA profile databases.

- MDSplus, two disruption-py retrievals per shot (dispy_methods.py):
  the 0D signals, and the full GEQDSK block of the shot's DISPY EFIT, the 1 kHz disruption-efit.
- IDA (integrated data analysis), one file per shot in a priority list of databases (ida.py):
  the Te and ne profiles, already GP fit on IDA's psi_N points.
  The ida fit method carries them onto rho_tor_norm through the DISPY q profile, nothing is refit.
TCV and DIII-D data has no release permission, so the datasets stop at the internal store (publishable False).
The sources sit on the DIII-D servers and /fusion/projects, so the unprocessed stage runs on omega.
OMEGA's system MDSplus cannot import under numpy 2, where disruption-py falls back to the mdsthin thin client by itself.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_validation_datasets.dispy_utils import passive_log_settings
from transport_validation_datasets.gp_fitting.batch_io import ShotFitInput
from transport_validation_datasets.machine.d3d.dispy_methods import (
    R0,
    D3DMethods,
    DispyEfitNicknameSetting,
    Uniform1kHzTimeSetting,
)
from transport_validation_datasets.machine.d3d.ida import (
    IdaDatabase,
    find_ida_path,
    find_ida_shots,
    ida_databases,
    ida_dataset,
)
from transport_validation_datasets.machine.generic import (
    channel_fit_rows,
    channel_rows_at_times,
    nearest_usable_reconstructions,
    rho_tor_norm_from_psi_n,
    snap_to_grid,
)
from transport_validation_datasets.store_schema import apply_signal_attrs
from transport_validation_datasets.workflow import DataWorkflow, DeviceSettings

# The 0D retrieval, custom methods only (dispy_methods.D3DMethods), each placed on the timebase causally
ZERO_D_RUN_METHODS = [
    "get_plasma_current",  # ip
    "get_toroidal_field",  # b0
    "get_efit_scalars",  # wmhd, beta_n
    "get_line_average_density",  # n_e_line_average
    "get_ohmic_power",  # p_ohm
    "get_radiated_power",  # p_rad
    "get_heating_powers",  # p_nbi, p_ech
    "get_boundary_parameters",  # aminor, rsurf, kappa, tritop, tribot
]
# Standardized name -> disruption-py column, SI. ip and b0 keep their source sign.
ZERO_D_SOURCES = {
    "ip": "ip",
    "b0": "b0",
    "energy_mhd": "wmhd",
    "beta_tor_norm": "beta_n",
    "n_e_line_average": "n_e_line_average",
    "minor_radius": "aminor",
    "geometric_axis_r": "rsurf",
    "elongation": "kappa",
    "triangularity_upper": "tritop",
    "triangularity_lower": "tribot",
    "power_ohm": "p_ohm",
    "power_radiated": "p_rad",
    "power_nbi": "p_nbi",
    "power_ec": "p_ech",
}

# An IDA slice whose Te error at the axis exceeds this fraction of Te there is an unconstrained core fit and is dropped.
# Ramp-up IDA-lite fits reach Te(0) = 47 +- 30 keV at n_e_line 5e18 (204180).
MAX_CORE_TE_RELATIVE_ERROR = 0.5

# The IDA databases, searched in priority order: a shot is read from the first database that has a file for it,
# and patterns may hold * wildcards. A database with a shotlist (a file in machine/d3d) only serves the shots on it.
# The union across databases is the default shotlist.
# The first two are full IDA, the rest IDA-lite (fewer diagnostics), and the results are very similar:
# the TMDB_V1c fits match HBP_database to a few percent on their 38 shared shots.
# The general-purpose databases hold thousands of shots each and are taken for HBP shots only.
DEFAULT_IDA_DATABASES = [
    {"pattern": "/fusion/projects/results/ida-results/HBP_database/IDA_{shot}_.cdf"},
    {
        "pattern": "/fusion/projects/xpsi/nelsonand/VVUQ/Output_KumarJul9/IDA_{shot}_*_.cdf"
    },
    {
        "pattern": "/fusion/projects/results/ida-results/TMDB_V1c/Output/IDA_{shot}_.cdf",
        "shotlist": "HBP_shotlist_2013_2025",
    },
    {
        "pattern": "/fusion/projects/results/ida-results/TMDB_V1a/Output/IDA_{shot}_.cdf",
        "shotlist": "HBP_shotlist_2013_2025",
    },
    {
        "pattern": "/fusion/projects/results/ida-results/TokaMaker_database/IDA_{shot}_.cdf",
        "shotlist": "HBP_shotlist_2013_2025",
    },
]

# Per-variable attributes. The store signals take their units and refs from STORE_SIGNAL_ATTRS.
SIGNAL_ATTRS = {
    "ip": {"description": "Measured plasma current, signed (PTDATA ip)"},
    "b0": {
        "description": "Vacuum toroidal field at r0, mu0 144 bcoil / (2 pi r0) from the TF coil current (PTDATA bcoil)",
    },
    "r0": {"description": "Reference major radius b0 is given at, the EFIT rzero"},
    "energy_mhd": {"description": "Stored energy from the 1 kHz DISPY EFIT (wmhd)"},
    "beta_tor_norm": {
        "description": (
            "Normalized toroidal beta as IMAS defines it, 100 beta_tor aminor |bcentr| / |ipmhd|[MA] with beta_tor = 2 mu0 <p> / bcentr^2, "
            "<p> = 2 wmhd / (3 volume) of the 1 kHz DISPY EFIT and bcentr the vacuum field at rcentr = r0, "
            "not the tree betan, which normalizes with the vacuum field at rout"
        ),
    },
    "n_e_line_average": {
        "description": "Line-averaged electron density from the DISPY EFIT tree (density), else the PCS estimate (PTDATA dssdenest)",
    },
    "minor_radius": {
        "description": "Minor radius of the plasma boundary, DISPY EFIT aminor"
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary, DISPY EFIT rsurf",
    },
    "elongation": {
        "description": "Elongation of the plasma boundary, DISPY EFIT kappa"
    },
    "triangularity_upper": {
        "description": "Upper triangularity of the plasma boundary, DISPY EFIT tritop",
    },
    "triangularity_lower": {
        "description": "Lower triangularity of the plasma boundary, DISPY EFIT tribot",
    },
    "power_ohm": {
        "description": (
            "Ohmic heating power from the 1 kHz DISPY EFIT (poh = Ip V_surf - dW_pol/dt), "
            "its derivatives centered least-squares slopes over +-100 ms (non-causal), clipped at 0"
        ),
    },
    "power_radiated": {
        "description": (
            "Total radiated power including the divertor, bolometer analysis prad_tot (4 ms), "
            "smoothed by a centered 50 ms boxcar applied twice to the raw channels (non-causal), clipped at 0"
        ),
    },
    "power_nbi": {"description": "Neutral beam power injected into the vessel (pinj)"},
    "power_ic": {
        "description": "Ion cyclotron heating power, zero (fast wave unused in these campaigns)",
    },
    "power_lh": {
        "description": "Lower hybrid heating power, zero (DIII-D has no LHCD)"
    },
    "power_ec": {
        "description": "Electron cyclotron power injected into the vessel (echpwrc)"
    },
    "ida_psi_n": {
        "description": "Normalized poloidal flux of the IDA profile points, of IDA's own reconstruction",
        "units": "dimensionless",
    },
    "ida_t_e": {
        "description": "IDA electron temperature fit on its psi_n points",
        "units": "eV",
    },
    "ida_t_e_error": {
        "description": "1-sigma uncertainty of the IDA electron temperature fit",
        "units": "eV",
    },
    "ida_n_e": {
        "description": "IDA electron density fit on its psi_n points",
        "units": "m^-3",
    },
    "ida_n_e_error": {
        "description": "1-sigma uncertainty of the IDA electron density fit",
        "units": "m^-3",
    },
}


@dataclass(frozen=True)
class D3DSettings(DeviceSettings):
    """DIII-D settings, the [d3d] table of the config file.

    Attributes:
        runtag: code_rundb runtag of the EFIT runs every EFIT signal comes from, the 1 kHz disruption-efit.
            A shot without a run under it is skipped.
        ida_databases: The IDA databases in priority order, one {"pattern", "shotlist"} table each,
            shotlist optional (see DEFAULT_IDA_DATABASES).
    """

    runtag: str = "DISPY"
    ida_databases: list = field(default_factory=lambda: list(DEFAULT_IDA_DATABASES))


class D3DDataWorkflow(DataWorkflow):
    """DIII-D specific data workflow for creating and processing datasets."""

    settings_cls = D3DSettings
    signal_attrs = SIGNAL_ATTRS
    # The profiles are IDA's own GP fits, carried onto the fit grid rather than refit
    fit_methods = ("ida",)
    prefit_profiles = True
    # No release permission for DIII-D data yet
    publishable = False

    min_pulse_length = 0.5
    min_filter = {
        "ip": 2e5,
        "energy_mhd": 1e4,
        # Ramp-ups reach 1e18 (199121), interferometer dropouts read below 3e17 (203836, 204188),
        # and a lost fringe count drives it negative (199126, down to -1.7e20)
        "n_e_line_average": 5e17,
    }
    max_filter = {"greenwald_fraction": 2.0}
    # The EFIT P_oh stays below 0.6 MW through the ramp-up and above 2 MW only at disruptive terminations.
    # P_rad reaches 16.0 MW in the iteration_3 store, so 17 MW only catches collapses.
    transient_filter = {"power_ohm": 2e6, "power_radiated": 17e6}
    # ip reads near 0 well past the end of the plasma, so the end is the last ip above its threshold.
    # A disruption's thermal quench can land ~60 ms before that end (199122),
    # and the EFIT P_oh smooths the quench out of reach of its transient filter.
    end_margin = 0.1
    min_radiated_fraction = 0.025
    # More radiated than put in, the same physical ceiling as the other devices (not checked on DIII-D data)
    max_radiated_fraction = 1.0
    density_ratio_bounds = (0.7, 1.3)
    # 203549, 203551, 203554: failed shots of run 20250529 (density limit with HFS pellets).
    # The beams did not fire ("No beams" in the logbook), so they are ohmic, pellet-perturbed, and 203554 disrupts.
    shot_blacklist = [203549, 203551, 203554]

    @cached_property
    def ida_databases(self) -> list[IdaDatabase]:
        """The IDA databases of the settings, in priority order."""
        return ida_databases(self.settings.ida_databases)

    def get_shotlist_from_source(self) -> list[int]:
        """Union of the shots every IDA database serves.

        Returns:
            The shots, sorted.

        Raises:
            FileNotFoundError: If no database has a file.
        """
        shots = find_ida_shots(self.ida_databases)
        if not shots:
            raise FileNotFoundError("No IDA files found in any configured IDA database")
        return shots

    def get_source_dataset(self, shot: int) -> xr.Dataset | None:
        """Read one shot from MDSplus and its IDA file into standardized signals.

        The 0D signals are placed causally (signal_on_grid),
        while the GEQDSK block and the IDA slices are snapped to the nearest grid time without interpolation.
        Both retrievals read the DISPY EFIT tree, so a shot without a DISPY run fails both.

        Args:
            shot: Shot number to read.

        Returns:
            The standardized dataset, or None when the shot cannot be built.
        """
        ida_path = find_ida_path(shot, self.ida_databases)
        if ida_path is None:
            self.record_failed_shot(shot, "No IDA file in any database.")
            return None
        ds_0d = _get_zero_d_dataset(shot, self.settings.runtag)
        if ds_0d is None:
            self.record_failed_shot(
                shot,
                f"No 0D data, no EFIT run under runtag {self.settings.runtag} or none at 1 kHz.",
            )
            return None
        ds_equilibrium = _get_geqdsk_dataset(shot, self.settings.runtag)
        # A missing GEQDSK node leaves NaN columns and no COCOS number
        if ds_equilibrium is None or "cocos" not in ds_equilibrium.attrs:
            self.record_failed_shot(shot, "No GEQDSK reconstruction.")
            return None

        timebase = ds_0d["time"].values
        ds_ida_native = ida_dataset(ida_path, shot)
        ds_ida_snapped = snap_to_grid(ds_ida_native, timebase)
        ds_ida = ds_ida_snapped.set_index(idx=["shot", "time"]).unstack("idx")
        if not ds_ida["ida_t_e"].notnull().any():
            self.record_failed_shot(shot, "No IDA slice inside the EFIT time range.")
            return None

        ds = xr.merge(
            [ds_0d, ds_equilibrium, ds_ida], compat="no_conflicts", join="outer"
        )
        rename = {
            source: name for name, source in ZERO_D_SOURCES.items() if source != name
        }
        ds = ds.rename(rename)
        # DIII-D's ICRF is unused in these campaigns and it has no lower hybrid, zero where ip is valid
        ds["power_ic"] = ds["ip"] * 0.0
        ds["power_lh"] = ds["ip"] * 0.0
        ds.attrs = {
            "cocos": ds_equilibrium.attrs["cocos"],
            # Per shot like cocos, the stack stage stores it as the r0 variable
            "r0": R0,
            "efit_runtag": self.settings.runtag,
            "ida_database": str(ida_path.parent),
        }
        apply_signal_attrs(ds, SIGNAL_ATTRS)
        return ds

    def prepare_fit_input(self, shot: int, ds: xr.Dataset) -> ShotFitInput | None:
        """Carry one shot's IDA slices onto rho_tor_norm for the ida fit method.

        1: Map each slice's psi_N points onto rho_tor_norm through the q profile of the nearest usable reconstruction
        2: Convert to the fit units (Te [keV], ne [1e20 m^-3])
        3: Drop the slices whose Te error at the axis exceeds MAX_CORE_TE_RELATIVE_ERROR of Te there,
           an unconstrained core fit

        IDA's psi_N comes from its own reconstruction, which the files do not name, while q comes from the DISPY EFIT.

        Args:
            shot: Shot number being staged.
            ds: The shot's unprocessed dataset.

        Returns:
            The fit input, or None when the shot has nothing fittable.
        """
        ds_shot = ds.squeeze("shot", drop=True)
        has_slice = ds_shot["ida_t_e"].notnull().any("ida_point").values
        slice_times = ds_shot["time"].values[has_slice]
        if slice_times.size == 0:
            logger.warning(f"Shot {shot}: no IDA slices to carry over")
            return None

        eq_index = nearest_usable_reconstructions(ds_shot, slice_times)
        psi_n_rows = channel_rows_at_times(ds_shot["ida_psi_n"], slice_times)
        qpsi = ds_shot["qpsi"].transpose("time", "psi_idx").values
        rho_tor_norm = np.full(psi_n_rows.shape, np.nan)
        for i_slice, i_eq in enumerate(eq_index):
            if i_eq < 0:
                continue
            rho_tor_norm[i_slice] = rho_tor_norm_from_psi_n(
                psi_n_rows[i_slice], qpsi[i_eq], self.settings.sol_extension
            )
        n_unmapped = int((eq_index < 0).sum())
        if n_unmapped:
            logger.debug(
                f"Shot {shot}: no usable equilibrium at {n_unmapped} of {eq_index.size} IDA slices"
            )

        te_y, te_err, ne_y, ne_err = channel_fit_rows(
            ds_shot, slice_times, prefix="ida"
        )
        psi_n_filled = np.where(np.isfinite(psi_n_rows), psi_n_rows, np.inf)
        i_axis = np.argmin(psi_n_filled, axis=1)
        slice_rows = np.arange(slice_times.size)
        te_axis = te_y[slice_rows, i_axis]
        te_error_axis = te_err[slice_rows, i_axis]
        with np.errstate(invalid="ignore"):
            unconstrained_core = te_error_axis > MAX_CORE_TE_RELATIVE_ERROR * te_axis
        te_y[unconstrained_core] = np.nan
        ne_y[unconstrained_core] = np.nan
        n_unconstrained = int(unconstrained_core.sum())
        if n_unconstrained:
            logger.info(
                f"Shot {shot}: dropped {n_unconstrained} IDA slices with a Te error at the axis over "
                f"{MAX_CORE_TE_RELATIVE_ERROR:g}x Te there"
            )

        fit_input = ShotFitInput(
            x=rho_tor_norm,
            te_y=te_y,
            te_err=te_err,
            ne_y=ne_y,
            ne_err=ne_err,
            time=slice_times,
        )
        if not fit_input.has_fittable_points():
            logger.warning(f"Shot {shot}: no mapped IDA points to carry over")
            return None
        return fit_input

    def fit_plot_channel_groups(
        self, shot: int, fit_input: ShotFitInput
    ) -> list | None:
        """Label the plotted points as IDA's, its fit on its own psi_N points.

        Args:
            shot: Shot number being plotted.
            fit_input: The shot's staged fit input.

        Returns:
            One (mask, color, label) triple over every point.
        """
        mask_all = np.ones(fit_input.x.shape[1], dtype=bool)
        return [(mask_all, "tab:blue", "IDA fit points")]


def _retrieve(
    shot: int, runtag: str, run_methods: list[str], custom_methods: list
) -> xr.Dataset | None:
    """Run one disruption-py retrieval of a shot on the DISPY EFIT and its 1 kHz timebase.

    Only the custom methods run, selected by name.
    Selected by column, the disruption-py built-ins serving the same columns would run too, and they interpolate.

    Args:
        shot: Shot number.
        runtag: code_rundb runtag of the EFIT run.
        run_methods: Names of the physics methods to run.
        custom_methods: The custom physics methods, or classes holding them.

    Returns:
        The retrieval on dims (shot, time), or None when disruption-py returned nothing.
    """
    retrieval_settings = RetrievalSettings(
        run_methods=run_methods,
        custom_physics_methods=custom_methods,
        efit_nickname_setting=DispyEfitNicknameSetting(runtag),
        time_setting=Uniform1kHzTimeSetting(),
        only_requested_columns=False,
    )
    result = get_shots_data(
        tokamak=Tokamak.D3D,
        shotlist_setting=[shot],
        retrieval_settings=retrieval_settings,
        output_setting=DatasetOutputSetting(path=False),
        log_settings=passive_log_settings(),
        num_processes=1,
    )
    if "shot" not in result or "time" not in result or result["time"].size == 0:
        return None
    return result.set_index(idx=["shot", "time"]).unstack("idx")


def _get_zero_d_dataset(shot: int, runtag: str) -> xr.Dataset | None:
    """The 0D signals of ZERO_D_RUN_METHODS, under the disruption-py column names.

    Args:
        shot: Shot number.
        runtag: code_rundb runtag of the EFIT run.

    Returns:
        The signals on dims (shot, time), or None when the retrieval failed.
    """
    return _retrieve(shot, runtag, ZERO_D_RUN_METHODS, [D3DMethods])


def _get_geqdsk_dataset(shot: int, runtag: str) -> xr.Dataset | None:
    """The GEQDSK block of the DISPY EFIT (D3DMethods.get_geqdsk_parameters).

    A retrieval of its own, since the COCOS number rides on the block's attributes.

    Args:
        shot: Shot number.
        runtag: code_rundb runtag of the EFIT run.

    Returns:
        The block on dims (shot, time, ...), or None when the retrieval failed.
    """
    return _retrieve(
        shot, runtag, ["get_geqdsk_parameters"], [D3DMethods.get_geqdsk_parameters]
    )
