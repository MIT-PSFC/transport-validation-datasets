# transport-validation-datasets
Consolidated methods for generating datasets to validate transport codes and train hybrid physics models

# Dataset structure

| Signals | Description |  IMAS | C-Mod Source | MAST Source | DIII-D Source | TCV Source | 
| ------ | ------ |        ------      |     ------        |      ------         |       ------     |
| ip | Measured plasma current | /summary/global_quantities/ip/value | 
| B0
| a_minor
| R0
| kappa
| delta_top
| delta_bot
| ne20_line_avg
| betan
| Wtot_MJ | /equilibrium/time_slice(itime)/global_quantities/energy_mhd (total kinetic pressure, includes fast ions)
| -------|
| ------ |
| P_RAD
| P_OH
| P_NBI
| P_ECRH
| P_ICRH
| P_LH
| ----|
| ----- |
| ne20_rho
| Te_keV_rho
| fresh_profiles
| ---- |
| Equilibrium things to re-make an EQDSK? TBD |
| fresh_equilibria

# Workflow

1: Pull unprocessed data from source and filter down to regions of validity
2: Perform GP profile fitting
3: Assemble dataset

# Running

One device per invocation, through the CLI:

```bash
# Both stages, C-Mod (needs MDSplus access through disruption-py)
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
```

`--help` lists every flag, and the stages resume: rerunning the same command
picks up whatever is not on disk yet.

# Sources

| Device | Source | Notes |
| ------ | ------ | ----- |
| C-Mod | MDSplus through disruption-py | Needs to run somewhere with tree access |
| MAST | Level 2 Zarr store at https://s3.echo.stfc.ac.uk/mast/level2/shots | Public, anonymous, read in a thread pool (`--prepare_workers`) |
