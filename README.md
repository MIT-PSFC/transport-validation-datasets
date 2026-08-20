# transport-validation-datasets
Consolidated methods for generating datasets to validate transport codes and train hybrid physics models

# Dataset structure

Every device is reduced to one schema, listed under [Final dataset](#final-dataset).
Signal names are IMAS-like, units are SI, and each variable carries its IMAS
data dictionary path under its `ref` attribute in the stored files. The
per-device sources of each signal live next to their attributes in
`machine/mast/mast_dataset.py` (`SIGNAL_ATTRS`) and
`machine/cmod/dispy_methods.py`.

# Workflow

1: Pull unprocessed data from source and filter down to regions of validity
2: Perform GP profile fitting
3: Assemble dataset

# Final dataset

One Zarr store per device, at `<data_assembly_dir>/<ds_name>/dataset_full/<ds_name>.zarr`,
built by `DataWorkflow.assemble_final_dataset`. Shots are stacked along `shot`
and NaN padded along every other dimension, so shots of different lengths line
up. Only one shot is ever held in memory while it is built.

The timebase is the unprocessed data's uniform 1 kHz grid. `time_idx` is the
grid ordinal, so shots of different lengths pad to a common size, and the
`time` variable carries the times themselves.

Profiles arrive one per Thomson pulse and equilibria on the reconstruction
clock, both far slower than 1 kHz, so both are held forward over the grid times
that follow them and `fresh_profile` / `fresh_equilibrium` mark the grid times
that carry a sample of their own. A sample is held for at most
`MAX_HOLD_PERIODS` of its own sampling period, so nothing is carried across the
end of the shot or a stretch the filtering cut away. By default the slices
whose Te and ne fits did not both come back usable are ignored, as though the
shot had no Thomson pulse there.

| Group | Signals | Dimensions |
| ------ | ------ | ------ |
| 0D | ip, b0, energy_mhd, beta_tor_norm, n_e_line_average, minor_radius, geometric_axis_r, elongation, triangularity_upper/lower, power_ohm/radiated/nbi/ic/lh | (shot, time_idx) |
| Time | time, fresh_profile, fresh_equilibrium | (shot, time_idx) |
| Fitted profiles | t_e, n_e, their _error, _gradient, _gradient_error, _fit_status | (shot, time_idx, rho) |
| Equilibrium | the full GEQDSK block: psirz, fpol, pres, ffprime, pprime, qpsi, rbdry, zbdry, rlim, zlim, rmagx, zmagx, simagx, sibdry, bcentr, current, rcentr, rleft, rdim, zmid, zdim | (shot, time_idx, grid) |

A signal the device does not have comes through as NaN, so the devices share
one schema. Everything is float32, flags included, because the padding between
shots of different lengths is NaN.

# Running

One device per invocation, through the CLI:

```bash
# Every stage, C-Mod (needs MDSplus access through disruption-py)
uv run python -m transport_validation_datasets.cli cmod /path/to/data_assembly_dir

# 20 shots of MAST (public S3, works anywhere with internet), unprocessed data only
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --max_num_shots 20 --stage unprocessed

# GP fitting only, dispatched to a SLURM cluster
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --stage fit \
    --cluster_ssh_host orcd-login \
    --cluster_partitions "sched_mit_psfc_r8@8:00:00" \
    --cluster_remote_workdir /path/on/cluster \
    --cluster_venv /path/on/cluster/.venv

# Assemble the final Zarr store from what the first two stages left on disk
uv run python -m transport_validation_datasets.cli mast /path/to/data_assembly_dir \
    --stage assemble
```

`--help` lists every flag, and the stages resume: rerunning the same command
picks up whatever is not on disk yet. Assembly is the exception, it always
rebuilds the store.

# Sources

| Device | Source | Shotlist | Notes |
| ------ | ------ | -------- | ----- |
| C-Mod | MDSplus through disruption-py | 2016 campaign from the C-Mod SQL summary table (Ip above 100 kA, pulse above 0.5 s), kept only on days with blessed Thomson data | Needs to run somewhere with MDSplus tree access |
| MAST | Level 2 Zarr store at https://s3.echo.stfc.ac.uk/mast/level2/shots, plus two level 1 groups: EFM for the GEQDSK safety factor and AYC for the Thomson profiles | 1101 shots from the M8 and M9 campaigns, shipped with the package | Public, anonymous, read in a thread pool (`--prepare_workers`) |
| DIII-D | | | |
| TCV | | | |
