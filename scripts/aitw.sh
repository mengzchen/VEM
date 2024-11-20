#!/usr/bin/env bash
#SBATCH -A berzelius-2024-341
#SBATCH -t 0-6:0:0
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"
#SBATCH --output logs/aitw_1102_v1.log
#SBATCH --error logs/aitw_1102_v1.log

# data 
# SAS="?sv=2023-01-03&st=2024-10-22T07%3A10%3A56Z&se=2024-10-29T07%3A10%3A00Z&skoid=d42edb90-9b8e-4f54-aaa5-dc6c37cabd88&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-10-22T07%3A10%3A56Z&ske=2024-10-29T07%3A10%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxlf&sig=NF2lB1KdLZVbfgQONvXqwaQXCU6P5OTN7VHp5A%2F5zfI%3D"
# ./azcopy copy "https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani/data$SAS" ./ --recursive

# env
# conda create -n qwen2vl
# conda activate qwen2vl
# git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# cd LLaMA-Factory
# pip install -e ".[torch,metrics]"
# cd ../

# cmd
# i1 -A berzelius-2024-341 --gres=gpu:1  --time 3:00:00 --mail-type "BEGIN,END,FAIL" --mail-user "v-zhengjiani@microsoft.com"
# squeue -u x_wenyi -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"

# run
# llamafactory-cli train configs/qwen2vl_aitw_label_1102_v1.yaml
# llamafactory-cli export configs/qwen2vl_merge.yaml
# python3 infer.py