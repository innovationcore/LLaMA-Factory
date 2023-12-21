#!/bin/bash

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=22222

#

export NCCL_DEBUG=warn
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME="ibp"
export NCCL_PROTO=simple
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnccl.so"
export NCCL_DEBUG=info

export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

#ifconfig -a 

#echo "-------"

#exit

cd /cm/shared/data/
docker run --shm-size 10g --ulimit memlock=-1 --ulimit stack=67108864 --privileged --ipc=host --network host --gpus all -e LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnccl.so" -e NCCL_SOCKET_IFNAME="ibp" -e CUDA_LAUNCH_BLOCKING=0  -e NCCL_PROTO=simple -e LOCAL_RANK=$SLURM_LOCALID -e RANK=$SLURM_PROCID -e WORLD_SIZE=$SLURM_NTASKS -e NCCL_DEBUG=warn -e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$MASTER_PORT -v /cm/shared/data:/data -v /cm/shared/data/LLaMA-Factory:/workspace -v /cm/shared/data/basemodels:/workspace/basemodels -v /cm/shared/data/outputmodels:/workspace/outputmodels -v /cm/shared/data/datasets/:/workspace/data/datasets -e WANDB_MODE=offline --rm llmfactory-test "/workspace/train.sh"

