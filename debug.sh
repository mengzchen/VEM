#!/usr/bin/env bash

#SBATCH -A NAISS2024-5-401
#SBATCH -t 0-1:0:0
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes 1
#SBATCH --output logs/debug.log
#SBATCH --error logs/debug.log

module load virtualenv/20.24.6-GCCcore-13.2.0
source /mimer/NOBACKUP/groups/softenable-design/zhengjiani/vlam/bin/activate

# source ~/.bashrc
# which node
# python3 models/gradio/autogui_demo.py --model our_webshop

# module load virtualenv/20.24.6-GCCcore-13.2.0
# pip install setuptools --upgrade
# srun --account=NAISS2024-5-401 --gpus-per-node=A100:1 --cpus-per-gpu=16 --time 1:00:00 python3 models/gradio/autogui_demo.py --model our_webshop
python3 train_rl_qwen2vl.py --task general --eval True

# salloc --account=NAISS2024-5-401 --gpus-per-node=A100:1 --cpus-per-gpu=16 --time 5:00:00
