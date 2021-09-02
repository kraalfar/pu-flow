#!/bin/bash
#SBATCH --job-name=flow_run
#SBATCH --error=flow_run-%j.err
#SBATCH --output=flow_run-%j.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

module load Python/Anaconda_v10.2019
source deactivate
source activate gpu_env
srun python flow_run.py