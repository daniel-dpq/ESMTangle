#!/bin/bash
#SBATCH --job-name=esmknot
#SBATCH --partition=normal
#SBATCH --account=hggroup
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=4   
#SBATCH --cpus-per-task=16    
#SBATCH --mem=128G              
#SBATCH --gres=gpu:4            
#SBATCH --time=48:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm3

export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

torchrun --nnodes=1 --nproc_per_node=4 --master_port=10000 mlm_tuning.py