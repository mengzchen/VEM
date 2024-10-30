#!/usr/bin/env bash
#SBATCH -A berzelius-2023-341
#SBATCH -t 0-1:0:0
#SBATCH --gres gpu:1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"
#SBATCH --output logs/debug.log
#SBATCH --error logs/debug.log

# data 
# SAS="?sv=2023-01-03&st=2024-10-22T07%3A10%3A56Z&se=2024-10-29T07%3A10%3A00Z&skoid=d42edb90-9b8e-4f54-aaa5-dc6c37cabd88&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-10-22T07%3A10%3A56Z&ske=2024-10-29T07%3A10%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxlf&sig=NF2lB1KdLZVbfgQONvXqwaQXCU6P5OTN7VHp5A%2F5zfI%3D"
# ./azcopy copy "https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/junting/training_raw_LAM.zip$SAS" ./ --recursive
# ./azcopy ls "https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/$SAS"  

# env
# conda create -n qwen2vl
# conda activate qwen2vl
# git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory
# pip install -e ".[torch,metrics]"
# cd ../

# run
nvidia-smi
llamafactory-cli train configs/qwen2_vl_lora.yaml