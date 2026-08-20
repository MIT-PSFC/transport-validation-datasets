#!/bin/bash
# Script used to make 20-shot sample dataset for MAST

MAX_NUM_SHOTS=20

uv run python -m transport_validation_datasets.cli mast \
    /usr/local/mfe/ml_data_dump/TORAX/transport_validation_datasets \
    --ds_name mast_sample_$MAX_NUM_SHOTS \
    --max_num_shots $MAX_NUM_SHOTS \
    --stage all \
    --cluster_ssh_host orcd-login \
    --cluster_partitions sched_mit_psfc_r8@11:00:00 \
    --cluster_remote_workdir /home/zkeith/orcd/scratch/transport_validation_datasets \
    --cluster_venv /home/zkeith/orcd/scratch/tests/transport_validation_datasets/.venv \
    --cluster_max_jobs 20 \
    --cluster_shots_per_batch 1 \
    --cluster_cpus_per_job 32 \
    --cluster_mem 64G \
    --cluster_max_retries 2
