#!/bin/bash
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --job-name=WorkMem
#SBATCH --account=fku@h100
##SBATCH -A fku@cpu

##SBATCH --partition=gpu_p2
##SBATCH --partition=gpu_p4
#SBATCH -C h100

#SBATCH --cpus-per-task=10
##SBATCH --cpus-per-task=3
##SBATCH --cpus-per-task=6
##SBATCH --cpus-per-task=8

#SBATCH --hint=nomultithread

#!/bin/bash
#SBATCH --job-name=EgoNC
#SBATCH --account=fku@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00

#SBATCH --output=logs/gpu_job%j.out
#SBATCH --error=errors/gpu_job%j.err

module load arch/h100
module load pytorch-gpu/py3/2.6.0

export PYTHONPATH=/lustre/fswork/projects/rech/fku/uir17ua/python_libs:$PYTHONPATH
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

srun torchrun --standalone --nproc_per_node=4 \
    train.py config/train_gpt2_wmego.py
