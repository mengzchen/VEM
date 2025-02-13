#!/usr/bin/env bash

#SBATCH -A NAISS2024-5-401
#SBATCH -t 0-48:0:0
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes 1
#SBATCH --output logs/webshop.log
#SBATCH --error logs/webshop.log

source /mimer/NOBACKUP/groups/softenable-design/zhengjiani/vlam/bin/activate

# python3 train_rl_qwen2vl.py --task general

python3 train_rl_qwen2vl.py --task webshop
