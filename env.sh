#!/usr/bin/env bash

export blob_dir="https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani"
export SAS="?sv=2023-01-03&st=2024-12-12T03%3A00%3A11Z&se=2024-12-19T03%3A00%3A00Z&skoid=d42edb90-9b8e-4f54-aaa5-dc6c37cabd88&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-12-12T03%3A00%3A11Z&ske=2024-12-19T03%3A00%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=1nyJOTwV8sJmoHci%2BjBfmFJKLfiXQUa%2FBc7GvSK3DGI%3D"

# get data
sudo apt-get install azcopy

if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
    cd checkpoints
    azcopy copy "$blob_dir/checkpoints/Auto-UI-Base$SAS"./ --recursive
    azcopy copy "$blob_dir/checkpoints/Qwen2-VL-7B-Instruct$SAS"./ --recursive
    azcopy copy "$blob_dir/checkpoints/blip2-opt-2.7b$SAS"./ --recursive
    azcopy copy "$blob_dir/checkpoints/roberta-base$SAS"./ --recursive
    cd ../
fi

if [ ! -d "images" ]; then
    azcopy copy "$blob_dir/images$SAS"./ --recursive
fi


conda create -n digirl python==3.10 -y
conda activate digirl
pip install -r digirl_requirements.txt
sudo apt-get update
sudo apt-get install libglib2.0-0 -y

python /cosmos/zhengjiani/DPO/DPO_train.py