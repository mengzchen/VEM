#!/usr/bin/env bash
#SBATCH -A berzelius-2024-341
#SBATCH -t 0-24:0:0
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"
#SBATCH --output logs/policy_general_qwenvl.log
#SBATCH --error logs/policy_general_qwenvl.log

# export PATH=/home/x_wenyi/.conda/envs/digirl/bin:$PATH
python3 train_rl_qwen2vl.py 