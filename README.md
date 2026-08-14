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
