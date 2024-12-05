#!/usr/bin/env bash
#SBATCH -A berzelius-2024-341
#SBATCH -t 0-24:0:0
#SBATCH --gres gpu:1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "v-zhengjiani@microsoft.com"
#SBATCH --output logs/rl_1201.log
#SBATCH --error logs/rl_1201.log

# cmd
# i1 -A berzelius-2024-341 --gres=gpu:1  --time 3:00:00 --mail-type "BEGIN,END,FAIL" --mail-user "v-zhengjiani@microsoft.com"
# squeue -u x_wenyi -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"

# data
# blob: https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus
# SAS="?sv=2023-01-03&st=2024-12-04T04%3A42%3A00Z&se=2024-12-11T04%3A42%3A00Z&skoid=d42edb90-9b8e-4f54-aaa5-dc6c37cabd88&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-12-04T04%3A42%3A00Z&ske=2024-12-11T04%3A42%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=TAbL29g1ftjcByVb1c6fvm0xA3ie7860UtasRQ91iuc%3D"
# ./azcopy copy "https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani/models$SAS" ./ --recursive
# ./azcopy remove "https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani/data_preprocess$SAS" --recursive
# ./azcopy ls "https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani/data$SAS" | cut -d/ -f 1 | awk '!a[$0]++'

# make digirl env
# conda create -n digirl python==3.10
# source ~/.bachrc
# conda activate digirl
# pip install -r digirl_requirements.txt
# sudo apt-get install libglib2.0-0

# export PATH=/home/x_wenyi/.conda/envs/digirl/bin:$PATH
python3 train_rl.py --config-path=configs/ --config-name=digirl_online
