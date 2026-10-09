# transport-validation-datasets
Consolidated methods for generating datasets to validate transport codes and train hybrid physics models

# Dataset structure

Every device is reduced to one schema, listed under [Datasets](#datasets).
Signal names are IMAS-like, units are SI, and each variable carries its IMAS
data dictionary path under its `ref` attribute in the stored files, with the
documentation page under `url`. The per-device sources of each signal live
next to their attributes in the device modules
(`SIGNAL_ATTRS` of `machine/cmod/cmod_dataset.py` and `machine/mast/mast_dataset.py`).
The units and data dictionary paths every store shares are in `store_schema.py`,
and the GEQDSK block's attributes in `machine/generic.py` (`GEQDSK_SIGNAL_ATTRS`).
The stack stage brings every variable onto that convention.
Each unprocessed file records the COCOS index of its GEQDSK signals in a root
attribute `cocos`. The final dataset carries it as the per-shot variable `cocos`.
It is identified per shot from the signs of the reconstruction's own Ip, B0, psi and q (`cocos_from_signs`),
COCOS 7 or 1 on C-Mod with the field direction and 3 on MAST.
The psi signals are per radian on both devices,
while the IMAS nodes their `ref` points at hold the total flux in Wb, in COCOS 17.

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

# Datasets

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
`time_idx` is the grid dimension, so shots of different lengths pad to a common size,
and the `time` variable carries the times themselves.

Profiles are from Thomson scattering and equilibria from a reconstruction,
both of which can be slower than 1 kHz or missing for certain timesteps,
so they are held forward over the grid times that follow them
and `fresh_profile` / `fresh_equilibrium` mark the grid times that carry a sample of their own.
A profile is held for at most `PROFILE_MAX_HOLD` (100 ms) on every device, whatever its Thomson cadence,
so dropped slices are bridged and `fresh_profile` tells a fresh profile from a held one.
The GEQDSK block of a reconstruction
is held for at most `MAX_HOLD_PERIODS` (1.5) periods of the reconstruction clock, or `EQUILIBRIUM_HOLD_FLOOR` (10 ms) when that is longer,
so a few missing reconstructions do not cut the shot,
and nothing is carried across the end of the shot or a stretch the filtering cut away.
The 0D signals taken from the reconstruction (`EQUILIBRIUM_0D_SIGNALS`: energy_mhd, beta_tor_norm and the shape)
hold the last usable reconstruction until the next one, with no limit (`hold_from_usable_reconstructions`),
so they change exactly where `fresh_equilibrium` is 1.
b0 is not one of them, it is the vacuum field of the measured toroidal field coil current.

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

The heating powers are the launched powers (IMAS `power_launched_*`), not the absorbed ones.
A heating system the device does not have is zero, so the devices share one schema.
Everything is float32, flags included, because the padding between shots of different lengths is NaN.
The power signals (power_ohm/radiated/nbi/ic/lh/ec) are clipped at zero since source records often dip negative
(bolometer baseline drift, channel pickup), and no heating or radiated power is physically negative.

Every stored value is causal, no grid time draws on a later sample, with these exceptions:
power_ohm and power_radiated are smoothed by a centered 50 ms boxcar applied twice (`smoothed_power`),
the kernel of DIII-D's bolometer postprocessing.
The EFIT and Thomson slices are snapped to the nearest grid time (`snap_to_grid`), up to 0.5 ms early,
and a Thomson slice can map through a reconstruction up to `EQ_MATCH_MAX_PERIODS` later than it (see [Filtering](#filtering)).

Every other 0D signal is placed on the 1 kHz grid without interpolation (`signal_on_grid`).
One sampled faster than the grid is averaged over each grid step, grid time t taking the mean of (t - 1 ms, t].
One sampled slower is held forward from its last finite sample for at most `MAX_HOLD_PERIODS` of its own sampling period.
The 0D signals taken from the reconstruction are held as above instead.
Derivatives are backward differences.

b0 is the vacuum toroidal field at the fixed major radius r0, as IMAS defines it:
the magnetics btor at 0.66 m on C-Mod,
and mu0 24 I_TF / (2 pi r0) from the toroidal field coil current `amc/tf_current` on MAST, at EFIT's bvac_r of 1.0 m,
which is the field EFIT's bvac_val gives.
ip and b0 keep their source sign, the cocos variable records the convention.
beta_tor_norm is normalized as IMAS defines it, 100 beta_tor a |b0| / |Ip|[MA] with beta_tor = 2 mu0 <p> / b0^2 and b0 at r0.
Both devices build it the same way (`generic.normalized_beta`),
from the reconstruction's own stored energy and volume, <p> = 2 W / (3 V), with its own a, b0 and Ip:
EFIT wplasm, vout, aout, bcentr and cpasma on C-Mod, efm plasma_energy, plasma_volume, minor_radius, bvac_val and plasma_current_c on MAST.
Neither reconstruction's own betan is IMAS's:
C-Mod's EFIT betan takes |btaxp|, the total field at the magnetic axis,
and MAST's efm betan takes the vacuum field at the geometric axis.
energy_mhd on MAST is efm plasma_energy, 3/2 the volume integral of the reconstructed pressure.
power_ohm is Ip V_loop - dW_pol/dt, computed from the equilibrium reconstruction alone the same way on every device
(`DataWorkflow.add_equilibrium_signals`, `generic.ohmic_power_on_grid`):
V_loop = sigma_Bp 2 pi dpsi_boundary/dt is the loop voltage at the LCFS from the block's `sibdry`,
W_pol = (pi / mu0) int |grad psi|^2 / R dR dZ is the poloidal field energy inside the boundary from `psirz`
(`poloidal_field_energy`),
and Ip is the block's `current`, so the sign convention is the reconstruction's own.
Both derivatives are backward differences between consecutive usable reconstructions,
and the result is held like every signal taken from the reconstruction, then smoothed.
No measured loop voltage enters,
the C-Mod wall flux loop `mflux:v0` reads 35-55 percent off the LCFS voltage at flattop on some shots.
n_e_line_average is the IMAS line average, the interferometer line integral over the chord length inside the plasma:
on C-Mod the TCI chord 4 `nl_04` over EFIT's `rco2v` for that chord (49 to 61 cm over a shot, held like the EFIT 0D signals),
on MAST the level 2 summary `line_average_n_e`.

# Filtering

Each stage drops what it can judge from its own inputs.
The stores record every threshold in their `filters` attribute,
and the device classes hold them (`min_filter`, `max_filter`, `transient_filter` and the attributes below).

A reconstruction is usable (`usable_reconstructions`) when its axis and boundary psi are finite and meaningfully different,
every value of its psirz and qpsi is finite, and its q profile gives a reasonable Phi_N map (`phi_n_map`).
All three stages that touch the equilibrium use only the usable ones:
the unprocessed stage holds the reconstruction's 0D signals from them and starts each kept segment where one reaches (step 6 below),
the fit stage maps the Thomson channels through the nearest one,
before or after the slice, within `EQ_MATCH_MAX_PERIODS` (1.5) periods of the reconstruction clock,
and the stack stage holds them onto the grid, so an unusable one is held over by the one before and is not marked fresh.
The reach and the hold both run on the reconstruction clock (`reconstruction_clock_period`), which counts the unusable ones too.

Unprocessed stage (`make_unprocessed_data_files`, `filter_and_plot`), per shot.
Steps 3 to 5 and 9 are the filter spec every device store shares (`filters.py`).
Every check from 3 to 5 cuts the grid times it fails out as a gap (`slice_filter_mask`).
A failed grid time before the end-of-shot cut also cuts the 50 ms before it (`FAILURE_MARGIN`, one `POWER_SMOOTHING_WINDOW`),
since the smoothed power_ohm and power_radiated carry the event that ends a segment one smoothing window ahead of it:

1. A shot in `shot_blacklist` or numbered below `first_shot` is skipped before its source is read (`excluded_shot_reason`).
2. The reconstruction's 0D signals are held and power_ohm is derived (`add_equilibrium_signals`).
3. End of shot (`end_of_shot_index`): the plasma ends at the last grid time with |ip| at or above its `min_filter` threshold,
   and everything after `end_margin` (50 ms, one smoothing window) before that is cut.
   A shot whose |ip| never reaches the threshold is rejected.
4. Every 0D signal must be finite, every `min_filter` signal at or above its threshold
   (ip compared as |ip|), and every `max_filter` signal at or below it, on the raw samples.
   The max filters take `greenwald_fraction` = n_e_line_average / n_GW with n_GW = Ip / (pi a^2),
   derived for the filter and not stored.
5. Grid times where a `transient_filter` signal, smoothed by a centered 5 ms boxcar, is above its threshold.
   The centered window only selects grid times, no stored value is smoothed by it.
   The unprocessed plots shade the transients red (`machine/plots.plot_unprocessed_data`).
6. The leading grid times of each segment are cut up to its first sample
   that a usable reconstruction of the same segment reaches within the GEQDSK block's hold,
   since the store would have no equilibrium before it.
   This is mostly the early parts of a shot, before its first usable reconstruction.
7. Only the longest segment is kept, shaded green in the accepted-shot plots.
   Only the kept segment's reconstructions reach the store,
   so step 6 trims each segment as if it alone were kept, and the longest is chosen after every trim.
8. The shot is rejected when the kept segment is shorter than `min_pulse_length` (C-Mod 0.5 s, MAST 0.2 s).
9. The shot is rejected when `shot_rejection_reason` finds a broken record in what is kept:
   - a mean `power_radiated` below `min_radiated_fraction` of the mean input power, ohmic plus auxiliary (a dead bolometer),
     or above `max_radiated_fraction` of it, since more cannot be radiated than is put in (`radiated_fraction_reason`)
   - a negative median `power_ohm`, the reconstruction's current and boundary flux disagreeing in sign (`ohmic_power_sign_reason`)
   - an `energy_mhd` rise from the first kept time to its peak greater than all the input power integrated to that time (`energy_sanity_reason`)

The standardized source pull of every shot is kept unfiltered in `01_unprocessed/source/`.
A rerun filters from it without touching the source.
Each pull is stamped with the code that read it (`pull_*` attributes, `pull_provenance`),
which the unprocessed file and the store carry, and the store lists the commits when its shots disagree.
A rerun does not check them, so delete `01_unprocessed/source/` after a change to a device read.
A rejection's note in `01_unprocessed/failed_shots/` records the filter settings that made it,
so a rerun skips a shot the same settings rejected and filters it again when they change.
Deleting the unprocessed files (`01_unprocessed/*.nc`) reruns a filter change on the shots that passed.
A change to the filter code alone needs the notes deleted too.
A shot that is accepted loses its note, so the notes count only the current rejections.

Fit stage: the Thomson channels map through the nearest usable reconstruction in reach,
the Thomson screens in `cleaning.py` run on every sample before fitting,
and the fit method's own checks give each slice a fit status.
A channel below psi_N 1 more than 5 mm outside its reconstruction's boundary contour sits in a private flux region,
under an X-point, so it is left unmapped.

Stack stage (`_internal_shot_dataset`), per shot:

1. The shots excluded or rejected by the unprocessed stage checks above are left out again.
2. A slice is dropped (`_usable_slice_mask`) when
   - its Te or ne fit status is not usable (not OK or REPAIRED)
   - Te at rho_tor_norm 1 is above `FLAT_TE_EDGE_RATIO` (0.4) of its peak,
     a flat profile from inboard and outboard Thomson channels that disagree after mapping
   - the 1 sigma band of Te or ne inside the LCFS is wider than the profile's peak,
     one channel's huge error carried into the band

   The previous slice holds over a dropped one like over any gap.
3. The shot is dropped when its fitted density disagrees with the interferometer (`fit_rejection_reason`).
   The shot median over its slices of mean(n_e over rho_tor_norm 0-1) / `n_e_line_average`
   must sit inside the device's `density_ratio_bounds`.
   A shot with no slice to compare is dropped too.
   The ratio is a proxy for the chord integral, and the bounds absorb its offset on each device.
4. The usable reconstructions are held onto the grid, as above.

Fit slices outside the unprocessed file's time span were fit before a filter change cut them and are dropped,
so a filter change restacks on the old fits.
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
  e.g. 24623 at 0.29-0.33 s (39 reconstructions in 26 shots out of ~1000).
- **Equilibrium gaps.** The 10 ms hold floor bridges a missing reconstruction in the GEQDSK block,
  but not two in a row on MAST (a 15 ms step), which leaves the block NaN there.
  The 0D signals taken from the reconstruction hold across any gap.
  A Thomson slice inside a bridged gap maps through no reconstruction (it reaches only `EQ_MATCH_MAX_PERIODS`),
  so the profile before it is held there.
- **power_ohm is the reconstruction's.** Its loop voltage is the boundary flux derivative of the equilibrium, not a flux loop,
  so it carries the reconstruction's flux noise (smoothed over 50 ms).
- **MAST summary signals.** The level 2 summary group is FAIR-MAST's own 1 kHz resampling of ip, n_e_line_average, power_radiated and power_nbi,
  whose causality this package does not verify.
- **EFIT `pres` goes slightly negative near the edge**, in 60 percent of C-Mod slices, down to ~2 percent of the core pressure.
  It is an artifact of the EFIT basis functions.

# Running

One device at a time, through the CLI:

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

`--help` lists every flag, and the stages resume
so rerunning the same command picks up whatever is not on disk yet.
Stack and publish are the exceptions, each always rebuilds its store.
The fit stage plots every shot's fits to a PDF, about a minute per shot.
`--skip_fit_plots` leaves them out, and a later fit stage without it plots the shots that have no PDF yet.
`--max_fit_pages` caps the pages of each PDF.
A cluster fit pulls each job's SLURM log back into `<ds_name>/logs/fit_jobs/`.

# Configuration

`--config` takes one or more TOML files, comma separated.
These configs are for the cluster the fits are dispatched to and the device-specific settings.
The files are layered so later ones override earlier ones.
For example, to run on the ORCD cluster with some per-user modifications, do something like this:

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
ssh_host = "orcd-user"
partitions = "mit_preemptable@8:00:00"
remote_workdir = "/path/on/cluster"
venv_path = "/path/on/cluster/.venv"
```

Keys are dataclass field names:
`ClusterFitConfig` in `gp_fitting/dispatcher.py` for `[cluster]`,
the device workflow's `settings_cls` for `[cmod]` and `[mast]`
(`CModSettings` in `machine/cmod/cmod_dataset.py`, `MASTSettings` in `machine/mast/mast_dataset.py`).
A key left out keeps its default.
A key the dataclass does not have, or a table that is neither `cluster` nor a device, is an error.
Without a `[cluster]` table each batch is fit locally in a worker process on this machine's cores,
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

The anchors are staged into the fit batches,
so changing them needs `--clean_fit_state` or a new `--ds_name`.
The fit stage refuses batches staged under any other fit setting, the device's fit bounds and fit grid included.

Both device tables also take `pedestal_rho_tor_norm`, the pedestal location every fit uses, 1.0 by default.
zk places its kernel's length-scale transition there, and akho centers its mtanh there.
One value for both Te and ne. It is staged and checked like the anchors.

```toml
[mast]
pedestal_rho_tor_norm = 1.0
```

Both device tables also take `sol_extension`, how the Thomson channels outside the LCFS are placed in rho_tor_norm.
Inside the LCFS the normalized toroidal flux Phi_N is the integral of |q| over psi_N (`phi_n_map`).
Where q diverges at the LCFS of a diverted plasma,
the integral runs past the last finite surface through q = a - b ln(1 - psi_N), fit to the surfaces inside it.
q is undefined outside the LCFS, so Phi_N continues linearly in psi_N,
with the slope from psi_N 0.95 to 1 (`"secant"`, the default) or the slope at the LCFS (`"tangent"`).
The staged positions depend on it, so it is checked like the anchors,
and it is recorded as the `sol_extension` attribute of the fit files and the stores.
The IMAS export maps the fit grid back onto psi through the same extension.

```toml
[cmod]
sol_extension = "tangent"
```

The `[cmod]` table also takes `efit_trees`, the EFIT trees a shot is read from, in order of preference.
The default is only EFIT21 (1 kHz magnetics-only), so a shot EFIT21 fails on is skipped.
Adding ANALYSIS after it turns on pulling those shots from the ANALYSIS tree instead (50 Hz magnetics-only).
Every retrieval reads the EFIT tree at least for its timebase, so a shot takes the first tree that serves all of them.
A tree fails when it is missing or its reconstruction is missing a node,
and the unprocessed file records the tree it used as its `efit_tree` attribute.
A tree slower than the 1 kHz grid (ANALYSIS reconstructs every ~20 ms) has its EFIT 0D signals held like EFIT21's,
from the last usable reconstruction until the next,
and `fresh_equilibrium` marks grid times where a usable reconstruction exists.
Shots already recorded in `01_unprocessed/failed_shots/` are not retried,
so their records need deleting for a rebuild to try another tree.

```toml
# Opt in to the ANALYSIS fallback
[cmod]
efit_trees = ["EFIT21", "ANALYSIS"]
```

# Shotlists and time windows

`--shotlist_file` takes one of two formats:

- plain: one shot number per line, blank lines and lines starting with `#` skipped
- windowed: 
  a CSV whose header holds `shot` (or `pulse_no`), `t_start` and `t_end` [s],
  one row per window and a shot on as many rows as it has windows.
  Other columns are ignored.
  Windows of one shot may overlap, but no two may share a center (closer than 1 ms),
  because the center is where the averaged profile is labeled.

Without a shotlist file the device's own list is used
(C-Mod queries its SQL summary table, MAST reads the list shipped with the package).

The MAST list holds the M7-M9 shots whose stores carry every signal the workflow reads,
whose plasma current passes the ip gates for at least `min_pulse_length`,
and whose Thomson chord passes within rho_tor_norm 0.1 of the magnetic axis for over 80% of that window.
The CSV records why each shot was rejected, and a rerun with it retries only the shots that could not be read:

```bash
uv run python -m transport_validation_datasets.machine.mast.shotlist build scratch/mast_shotlist/mast_scan.csv
```

The unprocessed stage is the same in every case:
the whole shot is read, filtered, and written,
so the unprocessed files can be reused when the windows change.
Windows act on the fit and stack stages, in one of two modes:

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
| C-Mod | MDSplus through disruption-py | 2016 campaign from the C-Mod SQL summary table (Ip above 100 kA, pulse above 0.5 s), kept only on days with blessed Thomson data | Needs to run somewhere with MDSplus tree access. A shot needs both the core and the edge Thomson system, the edge samples placed on the core's laser pulses one by one. |
| MAST | Level 1 Zarr store at https://s3.echo.stfc.ac.uk/mast/level1/shots (efm for the equilibrium, ayc for the Thomson profiles, amc for the toroidal field coil current), plus the summary group of the level 2 store at https://s3.echo.stfc.ac.uk/mast/level2/shots | 1678 shots from the M7-M9 campaigns, shipped with the package and built by `machine/mast/shotlist.py` | Public, anonymous, read in a thread pool (`--prepare_workers`) |
| DIII-D | | | |
| TCV | | | |
