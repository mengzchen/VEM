#!/usr/bin/env bash

#SBATCH -A NAISS2024-5-401
#SBATCH -t 0-12:0:0
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A100:4
#SBATCH --nodes 1
#SBATCH --output logs/critic_general_negative.log
#SBATCH --error logs/critic_general_negative.log
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"

module load virtualenv/20.24.6-GCCcore-13.2.0
source /mimer/NOBACKUP/groups/softenable-design/zhengjiani/vlam/bin/activate

# run
llamafactory-cli train configs/critic/critic_train.yaml
# llamafactory-cli export configs/critic/qwen2vl_merge.yaml
# python3 eval_tools/qwen2vl_infer.py
# python /cosmos/zhangdi/DPO/DPO_train.py