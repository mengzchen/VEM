#!/usr/bin/env bash

#SBATCH -A NAISS2024-5-401
#SBATCH -t 0-12:0:0
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:4
#SBATCH --nodes 1
#SBATCH --output logs/critic_general_qwen2.5.log
#SBATCH --error logs/critic_general_qwen2.5.log
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"

module load virtualenv/20.24.6-GCCcore-13.2.0
source /mimer/NOBACKUP/groups/softenable-design/zhengjiani/vlam/bin/activate

# run
# DISABLE_VERSION_CHECK=1 llamafactory-cli train configs/critic/critic_qwen2.5_general.yaml
# DISABLE_VERSION_CHECK=1 llamafactory-cli export configs/critic/qwen2vl_merge.yaml
python3 eval_tools/qwen2vl_infer.py
