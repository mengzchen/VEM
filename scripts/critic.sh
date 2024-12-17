#!/usr/bin/env bash

# run
#llamafactory-cli train configs/critic_1211.yaml
# llamafactory-cli export configs/qwen2vl_merge.yaml
python3 eval_tools/qwen2vl_infer.py
#python /cosmos/zhangdi/DPO/DPO_train.py