#!/bin/bash
#SBATCH --chdir=/gpfswork/rech/iqu/uvm91ap
#SBATCH --job-name=symm_ramp_2x2
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=64
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --constraint=a100
#SBATCH --gres=gpu:8
#SBATCH --account=iqu@a100
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rajah.nutakki@polytechnique.edu
#SBATCH --signal=B:USR1@30
#SBATCH --output=/gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/%j.out
echo Job running on nodes: $SLURM_NODELIST
module purge
module load arch/a100
module load gcc/12.2.0 anaconda-py3 openmpi/4.1.5
conda activate amd_gpu_nocallback
export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu
cleanup_function()
{
if [ "$recursion_number" -gt 4 ]; then
  echo "Recursion limit (4) reached, exiting"
  exit 1
else
  echo "Time limit reached, resubmitting ${recursion_number}th sbatch..."
  ((recursion_number+=1))
  sbatch --export=ALL,recursion_number=$recursion_number /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/sbatch11.sh
 exit 0
fi
}
trap 'cleanup_function' USR1
echo "recursion $recursion_number started at $(date)"
echo pip freeze > /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/requirements.txt
pip freeze > /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/requirements.txt
echo 'python -u -O /lustre/fswork/projects/rech/iqu/uvm91ap/repos/netket_pro_nocallback/deepnets/optimization/run.py --L=6 --J 0.8 1.0 --n_blocks 12 --features 72 --expansion_factor=2 --downsample_factor=2 --kernel_width=2 --output_head=Vanilla --samples_per_rank=512 --chains_per_rank=512 --discard_fraction=0.0 --iters 2500 2500 2500 2500 --lr 0.01 0.01 0.01 0.01 --alpha 0.5 0.5 0.5 0.5 --diag_shift 0.01 0.01 0.01 0.01 --diag_shift_end 0.0001 0.0001 0.0001 0.0001 --r=1e-06 --chunk_size=256 --save_every=100 --symmetries=0 --symmetry_ramping=1 --momentum=0.9 --post_iters=50 --double_precision=1 --time_it=0 --show_progress=0 --checkpoint=1 --seed=92  --save_base /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/ >> /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/${SLURM_JOB_ID}.out &'
srun python -u -O /lustre/fswork/projects/rech/iqu/uvm91ap/repos/netket_pro_nocallback/deepnets/optimization/run.py --L=6 --J 0.8 1.0 --n_blocks 12 --features 72 --expansion_factor=2 --downsample_factor=2 --kernel_width=2 --output_head=Vanilla --samples_per_rank=512 --chains_per_rank=512 --discard_fraction=0.0 --iters 2500 2500 2500 2500 --lr 0.01 0.01 0.01 0.01 --alpha 0.5 0.5 0.5 0.5 --diag_shift 0.01 0.01 0.01 0.01 --diag_shift_end 0.0001 0.0001 0.0001 0.0001 --r=1e-06 --chunk_size=256 --save_every=100 --symmetries=0 --symmetry_ramping=1 --momentum=0.9 --post_iters=50 --double_precision=1 --time_it=0 --show_progress=0 --checkpoint=1 --seed=92  --save_base /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/ >> /gpfswork/rech/iqu/uvm91ap/projects/deepNQS/ConvNext/14_10_24/L=6/symm_ramp_2x2/11/${SLURM_JOB_ID}.out & 
wait
