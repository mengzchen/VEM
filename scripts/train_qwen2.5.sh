#!/usr/bin/env bash

#SBATCH -A NAISS2024-5-401
#SBATCH -t 0-48:0:0
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes 1
#SBATCH --output logs/policy_qwen2.5.log
#SBATCH --error logs/policy_qwen2.5.log

module load virtualenv/20.24.6-GCCcore-13.2.0
source /mimer/NOBACKUP/groups/softenable-design/zhengjiani/vlam/bin/activate

python3 train_autogui.py --task general --eval
