#!/usr/bin/env bash
#SBATCH -A berzelius-2024-341
#SBATCH -t 0-12:0:0
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"
#SBATCH --output logs/critic_shopping.log
#SBATCH --error logs/critic_shopping.log

# run
llamafactory-cli train configs/critic/critic_train.yaml
# llamafactory-cli export configs/critic/qwen2vl_merge.yaml
# python3 eval_tools/qwen2vl_infer.py
# python /cosmos/zhangdi/DPO/DPO_train.py