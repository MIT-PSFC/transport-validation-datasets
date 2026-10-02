# transport-validation-datasets
Consolidated methods for generating datasets to validate transport codes and train hybrid physics models

# Dataset structure

Every device is reduced to one schema, listed under [The datasets](#the-datasets).
Signal names are IMAS-like, units are SI, and each variable carries its IMAS
data dictionary path under its `ref` attribute in the stored files, with the
documentation page under `url`. The per-device sources of each signal live
next to their attributes in `machine/mast/mast_dataset.py` (`SIGNAL_ATTRS`)
and `machine/cmod/dispy_methods.py`. The GEQDSK block's attributes are shared
(`machine/generic.py`, `GEQDSK_SIGNAL_ATTRS`). The stack stage brings every
variable onto that convention.
Each unprocessed file records the COCOS index of its GEQDSK signals in a root
attribute `cocos`. The final dataset carries it as the per-shot variable `cocos`.
It is identified per shot from the signs of the reconstruction's own Ip, B0, psi and q (`cocos_from_signs`).
The psi signals are per radian for MAST and C-Mod from EFIT, while the IMAS nodes their `ref` points at hold the total flux in Wb, in COCOS 17.

Every file records where it came from in its root attributes.
The unprocessed files carry the source package that pulled
the shot (`source_package`, `source_version`, `source_url`,
`source_retrieval_time`) and this package as it was at the time
(`transport_validation_datasets_commit`, `_dirty`, `_branch`, `_url`,
`_version`, plus `dependency_versions`). The stores merge those over every
shot they hold (a key the shots disagree on becomes a JSON list of its
values), and add the build (`build_time`, `build_host`), this package's state
at build time, and the run configuration as JSON: `device_settings` (the
`[<device>]` table), `filters` (every threshold the unprocessed and stack
stages applied, see [Filtering](#filtering)), and `fit_settings` (the fit staging knobs).

# Workflow

1: Pull unprocessed data from source and filter down to regions of validity
2: Perform GP profile fitting
3: Stack shots into the internal dataset
4: Publish dataset, stripping the internal-only signals

# The datasets

Two Zarr stores per device, under `<data_assembly_dir>/<ds_name>/04_datasets/`:

- `<ds_name>_internal.zarr`, the store for internal use, carrying every signal
  below. Built by `DataWorkflow.stack_internal_dataset` (the stack stage).
- `<ds_name>_published.zarr`, the store for release, derived from the internal
  one with the device's `published_strip_signals` stripped out. Built by
  `DataWorkflow.publish_dataset` (the publish stage).

Shots are stacked along `shot` and NaN padded along every other dimension, so
shots of different lengths line up. Only one shot is ever held in memory while
the internal store is built, and the published store is streamed from it.

The timebase is the unprocessed data's uniform 1 kHz grid.
Each shot keeps one contiguous stretch of it (see [Filtering](#filtering)),
so `time_idx` counts grid steps from the shot's first kept time,
shots of different lengths pad to a common size,
and the `time` variable carries the times themselves.
A shotlist with time windows keeps only the grid times inside them, so there `time` jumps between windows.

Profiles arrive one per Thomson sample and equilibria on the reconstruction
clock, both far slower than 1 kHz, so both are held forward over the grid times
that follow them and `fresh_profile` / `fresh_equilibrium` mark the grid times
that carry a sample of their own. A sample is held for at most
`MAX_HOLD_PERIODS` of its own sampling period, so nothing is carried across the
end of the shot or a diagnostic dropping out.
An equilibrium reconstruction, and every 0D signal taken from one, is held for at least `EQUILIBRIUM_HOLD_FLOOR` (10 ms),
so a few missing reconstructions do not cut the shot. By default the slices
the stack stage screens out (see [Filtering](#filtering)) are ignored, as though the
shot had no Thomson sample there.

| Group | Signals | Dimensions |
| ------ | ------ | ------ |
| 0D | ip, b0, energy_mhd, beta_tor_norm, n_e_line_average, minor_radius, geometric_axis_r, elongation, triangularity_upper/lower, power_ohm/radiated/nbi/ic/lh/ec | (shot, time_idx) |
| Time | time, fresh_profile, fresh_equilibrium | (shot, time_idx) |
| Per shot | r0, cocos | (shot) |
| Fitted profiles | t_e, n_e, their _error, _gradient, _gradient_error, _fit_status | (shot, time_idx, rho_tor_norm) |
| Equilibrium | the full GEQDSK block: psirz, fpol, pres, ffprime, pprime, qpsi, rbdry, zbdry, rlim, zlim, rmagx, zmagx, simagx, sibdry, bcentr, current, rcentr, rleft, rdim, zmid, zdim | (shot, time_idx, grid) |
| Raw Thomson channels (internal store only) | ts_channel_r, ts_channel_z, ts_channel_t_e, ts_channel_n_e, their _error | (shot, time_idx, ts_channel) |

The profiles are fit on rho_tor_norm from 0 to 1.6, so every fit anchor is on the fit grid and in the fit plots.
The fit files, the stores and the IMAS export keep them out to rho_tor_norm 1.1.

A heating system the device does not have is zero, so the devices share
one schema. Everything is float32, flags included, because the padding between
shots of different lengths is NaN.
The power signals (power_ohm/radiated/nbi/ic/lh/ec) are clipped at zero since source records often dip negative
(bolometer baseline drift, channel pickup), and no heating or radiated power is physically negative.

Every stored value is causal, no grid time draws on a later sample,
with two exceptions.
power_ohm and power_radiated are smoothed by a centered 50 ms boxcar applied twice (`smoothed_power`),
the kernel DIII-D's bolometer total comes with, since unsmoothed both are noise-dominated at 1 kHz.
The EFIT and Thomson slices are snapped to the nearest grid time (`snap_to_grid`), up to 0.5 ms early.
A 0D signal is never interpolated onto the 1 kHz grid (`signal_on_grid`).
One sampled faster than the grid is averaged over each grid step, grid time t taking the mean of (t - 1 ms, t].
One sampled slower is held forward from its last finite sample for at most `MAX_HOLD_PERIODS` of its own sampling period,
or `EQUILIBRIUM_HOLD_FLOOR` (10 ms) when it comes from the equilibrium reconstruction and that is longer.
Derivatives are backward differences.
b0 is the vacuum toroidal field at the fixed major radius r0, as IMAS defines it,
the magnetics btor at 0.66 m on C-Mod and the EFIT bvac_val at bvac_r (1.0 m) on MAST.
ip and b0 keep their source sign, the cocos variable records the convention.
power_ohm is Ip V_loop - dW_pol/dt on both devices.
MAST reads it from ESM (`pphix`), C-Mod computes it (`CmodPowerMethods.get_ohmic_power`)
from the flux loop voltage and W_pol = mu0 R_geo li Ip^2 / 4, R_geo the geometric major radius (EFIT `rout`).

The C-Mod 0D signals outside EFIT are read by custom methods (`CmodPlasmaMethods`, `CmodPowerMethods`),
since the disruption-py built-ins interpolate.
Every one of their records is faster than the grid, so each grid time takes the mean of the preceding millisecond.
Sample periods, checked on 5 shots from 2012 to 2016:

| Signal | Node (tree) | Period [ms] |
| --- | --- | --- |
| ip, b0 | `\ip`, `\btor` (magnetics) | 0.2 from -0.1 to 2.2 s, 10 outside |
| n_e_line_average | `.tci.results:nl_04` (electrons) | 0.5 |
| power_radiated | `\twopi_diode` (spectroscopy) | 0.333 |
| power_ic | `\rf_power_net` (rf) | 0.1 |
| power_lh | `\top.results:netpow` (lh) | 0.04 |
| V_loop of power_ohm | `\top.mflux:v0` (analysis) | 0.2 |

power_ic and power_lh are 0 outside their records and on shots without the system.
power_radiated is NaN outside its record.

# Filtering

Each stage drops what it can judge from its own inputs.
The stores record every threshold in their `filters` attribute.

A reconstruction is usable (`usable_reconstructions`) when its axis and boundary psi are finite and meaningfully different,
and every value of its psirz and qpsi is finite.
All three stages that touch the equilibrium contain only the usable ones:
the unprocessed stage starts each kept segment where one reaches (step 5 below),
the fit stage maps the Thomson channels through the nearest one,
and the stack stage holds them onto the grid, so an unusable one is held over by the one before and is not marked fresh.
The reach and the hold both run on the reconstruction clock (`reconstruction_clock_period`), which counts the unusable ones too.

Unprocessed stage (`filter_and_plot`), per shot.
Steps 2 to 4 and 8 are the filter spec every device store shares (`filters.py`).
Every check from 2 to 4 cuts the grid times it fails out as a gap (`slice_filter_mask`):

1. A shot in `shot_blacklist` or numbered below `first_shot` is skipped before its source is read (`excluded_shot_reason`).
2. End of shot (`end_of_shot_index`): the plasma ends at the last grid time with |ip| at or above its `min_filter` threshold,
   and everything after `end_margin` (C-Mod 20 ms, MAST 40 ms) before that is cut.
   A shot whose |ip| never reaches the threshold is rejected.
3. Every 0D signal must be finite, every `min_filter` signal at or above its threshold
   (ip compared as |ip|), and every `max_filter` signal at or below it, on the raw samples.
   The max filters take `greenwald_fraction` = n_e_line_average / n_GW with n_GW = Ip / (pi a^2),
   derived for the filter and not stored.
4. Grid times where a `transient_filter` signal, smoothed by a centered 5 ms boxcar, is above its threshold.
   The centered window only selects grid times, no stored value is smoothed by it.
   The unprocessed plots shade the transients red.
5. The leading grid times of each segment are cut up to its first sample
   that a usable reconstruction (`usable_reconstructions`) of the same segment reaches
   within the hold (`MAX_HOLD_PERIODS` of the reconstruction clock, at least `EQUILIBRIUM_HOLD_FLOOR`), since the store would have no equilibrium before it.
   This is mostly the early parts of a shot, before its first usable reconstruction.
6. Only the longest segment is kept, shaded green in the accepted-shot plots.
   Only the kept segment's reconstructions reach the store,
   so step 5 trims each segment as if it alone were kept, and the longest is chosen after every trim.
7. The shot is rejected when the kept segment is shorter than `min_pulse_length` (C-Mod 0.5 s, MAST 0.2 s).
8. The shot is rejected when `shot_rejection_reason` finds a broken record in what is kept:
   - a mean `power_radiated` below `min_radiated_fraction` of the mean input power, ohmic plus auxiliary (a dead bolometer),
     1 percent on C-Mod and 2.5 percent on MAST (`radiated_fraction_reason`)
   - a sanity check for conservation of energy, triggered if `energy_mhd` rise from the first kept time to its peak is greater than all input power integrated to that time point (`energy_sanity_reason`).

| Threshold | C-Mod | MAST |
| --- | --- | --- |
| min ip | 100 kA | 210 kA |
| min energy_mhd | 2.7 kJ | 5 kJ |
| max greenwald_fraction | 2.0 | 2.0 |
| max power_radiated | | 4 MW |
| transient power_ohm | 5 MW | 5 MW |
| transient power_radiated | 5.5 MW | 3 MW |
| end_margin | 20 ms | 40 ms |
| min_pulse_length | 0.5 s | 0.2 s |
| min_radiated_fraction | 0.01 | 0.025 |
| density_ratio_bounds | 0.72-1.3 | 0.7-1.3 |

The standardized source pull of every shot is kept unfiltered in `01_unprocessed/source/`.
A rerun filters from it without touching the source.
A rejection's note in `01_unprocessed/failed_shots/` records the filter settings that made it,
so a rerun skips a shot the same settings rejected and filters it again when they change.
Deleting the unprocessed files (`01_unprocessed/*.nc`) reruns a filter change on the shots that passed.
A change to the filter code alone needs the notes deleted too.

Fit stage: the Thomson channels map through the nearest usable reconstruction in reach,
the Thomson screens in `cleaning.py` run on every sample before fitting,
and the fit method's own checks give each slice a fit status.

Stack stage (`_internal_shot_dataset`), per shot:

1. The shots excluded or rejected by the unprocessed stage checks above are left out again.
2. A slice is dropped (`usable_slice_mask`) when
   - its Te or ne fit status is not usable (not OK or REPAIRED)
   - Te at rho_tor_norm 1 is above `FLAT_TE_EDGE_RATIO` (0.4) of its peak,
     a flat profile from inboard and outboard Thomson channels that disagree after mapping
   - the 1 sigma band of Te or ne inside the LCFS is wider than the profile's peak,
     one channel's huge error carried into the band

   The previous slice holds over a dropped one like over any gap.
3. The shot is dropped when its fitted density disagrees with the interferometer (`fit_rejection_reason`).
   The shot median over its slices of mean(n_e over rho_tor_norm 0-1) / `n_e_line_average`
   must sit inside the device's `density_ratio_bounds` (C-Mod 0.72-1.3, MAST 0.7-1.3).
   A shot with no slice to compare is dropped too.
   The ratio is a proxy for the chord integral, and the bounds absorb its offset on each device.
4. The usable reconstructions are held onto the grid, as above.

`export_to_imas` runs checks 1-3 too (`_usable_shot_fit`), so the IMAS export holds the same shots as the stores.

MAST starts at shot 23809 (`first_shot`).
Before it the Thomson density reads ~0.87x the interferometer, against 0.98-1.00 after,
indicating a large change in the Thomson density calibration.

# Known limitations

- **Thomson against interferometer.** Inside the density bounds the two still differ shot to shot.
  The kept shots sit at 0.76-1.16 on C-Mod and 0.75-1.10 on MAST.
- **MAST transients inside the kept windows.** Reconnection events and Ip spikes that stay under the transient thresholds remain,
  e.g. 28203 at 0.343 s, where core Te drops from 0.55 to 0.12 keV, Ip spikes from 0.53 to 0.68 MA and P_rad reaches 2.8 MW.
  The transient filter needs P_rad above 3 MW after smoothing.
- **MAST EFIT vertical glitches.** Single reconstructions jump zmagx and zbdry by 5-10 cm and come back at the next one,
  e.g. 24623 at 0.29-0.33 s (though this is minor, 39 reconstructions in 26 shots out of ~1000 total shots).
- **Equilibrium gaps.** The 10 ms hold floor bridges a missing reconstruction on either device,
  but not two in a row on MAST (a 15 ms step), which still cuts every equilibrium signal.
  A Thomson slice inside a bridged gap maps through no reconstruction (it reaches only `EQ_MATCH_MAX_PERIODS`),
  so the profile before it is held there.
- **MAST `power_ohm` gaps.** Some converged reconstructions have no `pphix` (up to 25 ms at flat-top in 24891).
  The hold bridges up to 10 ms of them, and longer ones are cut out as gaps.
- **EFIT `pres` goes slightly negative near the edge**, in 60 percent of C-Mod slices, down to ~2 percent of the core pressure.
  It is an artifact of the EFIT basis functions.

# Running

One device per invocation, through the CLI:

```bash
# Every stage, C-Mod (needs MDSplus access through disruption-py)
uv run python -m transport_validation_datasets.cli cmod /path/to/data_assembly_dir

# 20 shots of MAST (public S3, works anywhere with internet), unprocessed data only
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --max_num_shots 20 --stage unprocessed

# GP fitting only, dispatched to the SLURM cluster in the config files
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --stage fit --config configs/orcd.toml,configs/$USER.user.toml

# Stack the internal Zarr store from what the first two stages left on disk
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --stage stack

# Derive the published store from the internal one
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --stage publish
```

`--help` lists every flag, and the stages resume: rerunning the same command
picks up whatever is not on disk yet. Stack and publish are the exceptions,
each always rebuilds its store.
The fit stage plots every shot's fits to a PDF, about a minute per shot.
`--skip_fit_plots` leaves them out. If a later fit stage is run without this flag it will plot the shots that have no PDF yet.

# Configuration

`--config` takes one or more TOML files, comma separated, holding what does
not change from run to run: the cluster the fits are dispatched to and the
device-specific settings. The files are layered, a later one overrides an
earlier one key by key. `configs/orcd.toml` is the shared file; the cluster
paths are per user and go in `configs/<user>.user.toml`, which git ignores:

```bash
--config configs/orcd.toml,configs/$USER.user.toml
```

```toml
# configs/orcd.toml
[cluster]
ssh_host = "orcd-login"
partitions = "sched_mit_psfc_r8@11:00:00"
max_concurrent_jobs = 20

[cmod]
efit_trees = ["EFIT21"]

# configs/<user>.user.toml
[cluster]
remote_workdir = "/path/on/cluster"
venv_path = "/path/on/cluster/.venv"
```

Keys are dataclass field names: `ClusterFitConfig` in `gp_fitting/dispatcher.py`
for `[cluster]`, the device workflow's `settings_cls` for `[cmod]` and `[mast]`
(`CModSettings` in `machine/cmod/cmod_dataset.py`, `MASTSettings` in `machine/mast/mast_dataset.py`).
A key left out keeps its default.
A key the dataclass does not have, or a table that is neither `cluster` nor a device, is an error.
Without a `[cluster]` table the fits run locally in the calling process, 
and without `--config` everything keeps its default.
Run-specific choices (`--ds_name`, `--shotlist_file`, `--stage`, `--method`, ...) stay command line flags.

Both device tables take the fit anchors, the virtual observations every fit method adds to every Thomson slice.
Each is a list of `[rho_tor_norm, value, error]` rows in the fit units, Te in keV and ne in 1e20 m^-3, gradients per unit rho_tor_norm.
The defaults pin each profile to zero value and gradient at rho_tor_norm 1.3 to 1.6, and zero gradient at the axis.
They sit past the SOL channels, which rho_tor_norm stretches out to ~1.25.
Moving an anchor past 1.6 needs a device fit grid that reaches it.

```toml
[mast]
te_value_anchors = [[1.3, 0.0, 0.01], [1.4, 0.0, 0.01], [1.5, 0.0, 0.01], [1.6, 0.0, 0.01]]
te_grad_anchors = [[0.0, 0.0, 0.1], [1.3, 0.0, 0.1], [1.4, 0.0, 0.1], [1.5, 0.0, 0.1], [1.6, 0.0, 0.1]]
# ne_value_anchors, ne_grad_anchors likewise
```

The anchors are staged into the fit batches, so changing them needs
`--clean_fit_state` or a new `--ds_name`.

Both device tables also take `pedestal_rho_tor_norm`, the pedestal location every fit uses, 1.0 by default.
zk places its kernel's length-scale transition there, and akho centers its mtanh there.
One value for both Te and ne. It is staged and checked like the anchors.

```toml
[mast]
pedestal_rho_tor_norm = 0.95
```

Both device tables also take `sol_extension`, how the Thomson channels outside the LCFS are placed in rho_tor_norm.
Inside the LCFS the normalized toroidal flux Phi_N is the integral of q over psi_N.
q is undefined outside it, so Phi_N continues linearly in psi_N,
with the slope from psi_N 0.95 to 1 (`"secant"`, the default) or the slope at the LCFS (`"tangent"`).
The staged positions depend on it, so it is checked like the anchors,
and it is recorded as the `sol_extension` attribute of the fit files and the stores.
The IMAS export maps the fit grid back onto psi through the same extension.

```toml
[cmod]
sol_extension = "tangent"
```

The `[cmod]` table also takes `efit_trees`, the EFIT trees a shot is read from, in order of preference.
The default is only EFIT21, so a shot EFIT21 fails on is skipped.
Adding ANALYSIS after it turns on pulling those shots from the ANALYSIS tree instead.
Every retrieval reads the EFIT tree at least for its timebase, so a shot takes the first tree that serves all of them.
A tree fails when it is missing or its reconstruction is missing a node,
and the unprocessed file records the tree it used as its `efit_tree` attribute.
A tree slower than the 1 kHz grid (ANALYSIS reconstructs every ~20 ms) has its EFIT 0D signals held forward onto the grid (`signal_on_grid`),
and `fresh_equilibrium` marks grid times where the reconstruction exists.
Shots already recorded in `01_unprocessed/failed_shots/` are not retried,
so their records need deleting for a rebuild to try another tree.

```toml
# Opt in to the ANALYSIS fallback
[cmod]
efit_trees = ["EFIT21", "ANALYSIS"]
```

# Shotlists and time windows

`--shotlist_file` takes one of two formats:

- plain: one shot number per line
- windowed: a CSV whose header holds `shot` (or `pulse_no`), `t_start` and
  `t_end` [s], one row per window and a shot on as many rows as it has windows.
  Other columns are ignored. Windows of one shot may overlap, but no two may
  share a center (closer than 1 ms), because the center is where the averaged profile
  is labeled.

Without a shotlist file the device's own list is used (C-Mod queries its SQL
summary table, MAST reads the list shipped with the package, which
`machine/mast/shotlist.py` builds from the FAIR-MAST catalog).

The unprocessed stage is the same in every case: the whole shot is read,
filtered, and written, so the unprocessed files can be reused when the windows
change. Windows act on the fit and stack stages, in one of two modes:

| Mode | Flags | Fit stage | Store |
| ---- | ----- | --------- | ----- |
| per sample | plain shotlist | every Thomson sample fit on its own | the whole accepted shot |
| windowed per sample | windowed shotlist | only the Thomson samples inside a window are fit, on their own | only the grid times inside the windows, profiles held forward but never across a window boundary |
| window average | windowed shotlist and `--average_windows` | every rho-mapped Thomson point inside a window is pooled and fit as one profile (a sample in overlapping windows goes into each) | only the grid times inside the windows, each window filled with its one profile, `fresh_profile` set at the grid time nearest the window center. Where windows overlap a grid time carries the profile of the holding window whose center is nearest |

`--average_windows` with a plain shotlist is an error. Every staged batch and
every fit result records the mode and the windows it was built with, and the
fit and stack stages compare them with the shotlist before doing anything. A
shot whose windows changed, a shot the shotlist no longer lists, or a mode
switch stops the run with an error, so an edited shotlist can never quietly
ship profiles fit for other windows. `--clean_fit_state` restages everything
(or build under a new `--ds_name`). The window list of each shot and the
window of each fitted row are kept in the `03_fit_results` files for debugging (`windows`
attribute, `window_index` coordinate), not in the final stores.

```bash
# C-Mod scenarios, one fit per Thomson sample inside each window
uv run python -m transport_validation_datasets.cli cmod /path/to/data_assembly_dir \
    --ds_name cmod_scenarios --shotlist_file scenarios.csv

# The same windows, one pooled fit per window
uv run python -m transport_validation_datasets.cli cmod /path/to/data_assembly_dir \
    --ds_name cmod_scenarios_avg --shotlist_file scenarios.csv --average_windows
```

# Sources

| Device | Source | Shotlist | Notes |
| ------ | ------ | -------- | ----- |
| C-Mod | MDSplus through disruption-py | 2016 campaign from the C-Mod SQL summary table (Ip above 100 kA, pulse above 0.5 s), kept only on days with blessed Thomson data | Needs to run somewhere with MDSplus tree access. |
| MAST | Level 1 Zarr store at https://s3.echo.stfc.ac.uk/mast/level1/shots: EFM for the equilibrium (GEQDSK and 0D), ESM for the ohmic power, AYC for the Thomson profiles. The level 2 store at https://s3.echo.stfc.ac.uk/mast/level2/shots supplies only the summary signals (ip, NBI and radiated power, line averaged density) | 1671 shots from the M7-M9 campaigns, shipped with the package and built by `machine/mast/shotlist.py` | Public, anonymous, read in a thread pool (`--prepare_workers`) |
| DIII-D | | | |
| TCV | | | |
