#!/usr/bin/env bash

# env
#git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
#cd LLaMA-Factory
#pip install -e ".[torch,metrics]"
#cd ../

# run
llamafactory-cli train configs/critic_1211.yaml
python /cosmos/zhangdi/DPO/DPO_train.py
# llamafactory-cli export configs/qwen2vl_merge.yaml
# python3 infer.py