#!/usr/bin/env bash

#SBATCH -A NAISS2024-5-401
#SBATCH -t 0-48:0:0
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:2
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes 1
#SBATCH --output logs/debug.log
#SBATCH --error logs/debug.log
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"

module load virtualenv/20.24.6-GCCcore-13.2.0
source /mimer/NOBACKUP/groups/softenable-design/zhengjiani/vlam/bin/activate

python3 train_seeclick.py --task general --eval

# salloc --account=NAISS2024-5-401 --gpus-per-node=A100:1 --cpus-per-gpu=16 --time 1:00:00
