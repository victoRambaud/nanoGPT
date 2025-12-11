#!/bin/bash
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --job-name=PRETRAIN_GPT2
#SBATCH --account=fku@h100
##SBATCH -A fku@cpu

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

#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00
#SBATCH --output=logs/gpu_job%j.out
#SBATCH --error=errors/gpu_job%j.out


#######################################################################################


module load arch/h100
module load pytorch-gpu/py3/2.6.0
export PYTHONPATH=/lustre/fswork/projects/rech/fku/uir17ua/python_libs:$PYTHONPATH


# source activate mfa
# set -x

# srun --ntasks=4 python -u train.py config/train_gpt2.py
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
# sbatch job.sh