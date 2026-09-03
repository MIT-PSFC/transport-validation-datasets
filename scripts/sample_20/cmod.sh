#!/bin/bash
# Script used to make 20-shot sample dataset for C-Mod
# NOTE: Expects bootstrap_remote.sh to have set up the cluster, and configs/$USER.user.toml to hold its paths

MAX_NUM_SHOTS=20

uv run python -m transport_validation_datasets.cli cmod \
    /usr/local/mfe/ml_data_dump/TORAX/transport_validation_datasets \
    --ds_name cmod_sample_$MAX_NUM_SHOTS \
    --max_num_shots $MAX_NUM_SHOTS \
    --stage all \
    --config "configs/orcd.toml,configs/$USER.user.toml"
