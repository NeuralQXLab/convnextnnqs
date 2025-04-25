#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=64
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --constraint=a100
#SBATCH --gres=gpu:8
#SBATCH --account=rgs@a100
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rajah.nutakki@polytechnique.edu

echo Job running on nodes: $SLURM_NODELIST
module purge
module load arch/a100
export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu
uv --project /lustre/fsn1/projects/rech/rgs/uvm91ap/uv_envs/nk_pro_update export >> requirements.txt
srun uv --project /lustre/fsn1/projects/rech/rgs/uvm91ap/uv_envs/nk_pro_update run /lustre/fswork/projects/rech/rgs/uvm91ap/repos/netket_pro_update/deepnets/scripts/../optimization/run.py --config /lustre/fswork/projects/rech/rgs/uvm91ap/projects/deepNQS/ConvNext/03_12_24/L10_2x2_symmramp/9/config.yaml --n_blocks 12 --seed 3 
