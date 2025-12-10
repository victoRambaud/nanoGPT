#!/bin/bash
#SBATCH --job-name=PRETRAIN
#SBATCH -A bxp@h100
##SBATCH -A bxp@v100
##SBATCH -A bxp@cpu

##SBATCH --partition=gpu_p2
##SBATCH --partition=gpu_p4
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=10
##SBATCH --cpus-per-task=3
##SBATCH --cpus-per-task=6
##SBATCH --cpus-per-task=8

#SBATCH --hint=nomultithread

#SBATCH --qos=qos_gpu_h100-dev 
#SBATCH --time=2:00:00
#SBATCH --output=logs/gpu_job%j.out
#SBATCH --error=errors/gpu_job%j.out


#######################################################################################


module load arch/h100
module load pytorch-gpu/py3/2.8.0
export PYTHONPATH=/lustre/fswork/projects/rech/bxp/ubm84dh/python_libs:$PYTHONPATH


# source activate mfa
# set -x

# srun --ntasks=4 python -u train.py config/train_gpt2.py
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
# sbatch job.sh