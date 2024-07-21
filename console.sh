#!/bin/bash

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,GRAPH

export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

export MPI_TYPE=pmix

export FORCE_TORCHRUN=1


# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

#/bin/bash -c set > /project/ibi-staff/llmfactory/set.log

export CLEARML_CONFIG_FILE=/root/clearml.conf

export CONTAINER=/project/ibi-staff/llmfactory/container/llmfactory.sqfs

export MOUNTS=/project/ibi-staff/llmfactory/custom_data:/app/custom_data,/project/ibi-staff/llmfactory/working_dir:/app/working_dir,/project/ibi-staff/llmfactory/config/clearml.conf:/root/clearml.conf,/project/i
bi-staff/llmfactory/config/hosts:/etc/hosts,/project/ibi-staff/llmfactory/models:/app/basemodels

srun --mpi=$MPI_TYPE --nodes=1 --gpus-per-node=1 -p priority --container-image=$CONTAINER --container-save=$CONTAINER --container-writable --no-container-mount-home --container-mounts=$MOUNTS --pty bash