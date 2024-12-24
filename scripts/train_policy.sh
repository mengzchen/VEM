#!/usr/bin/env bash
#SBATCH -A berzelius-2024-341
#SBATCH -t 0-12:0:0
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"
#SBATCH --output logs/policy.log
#SBATCH --error logs/policy.log

# export PATH=/home/x_wenyi/.conda/envs/digirl/bin:$PATH
python3 train_rl.py --config-path=configs/ --config-name=rl