#!/usr/bin/env bash

export blob_dir="https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani"
export SAS="?sv=2023-01-03&st=2024-12-16T05%3A48%3A43Z&se=2024-12-23T05%3A48%3A00Z&skoid=d42edb90-9b8e-4f54-aaa5-dc6c37cabd88&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2024-12-16T05%3A48%3A43Z&ske=2024-12-23T05%3A48%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=IQSgsRlQuAkhWjT68YxVSMNVxSg1RJ1bQO4wInJWhaI%3D"

#azcopy copy ./checkpoints/critic_1211_more "$blob_dir/checkpoints$SAS" --recursive

# get data
#sudo apt-get install azcopy
#
#if [ ! -d "checkpoints" ]; then
#    mkdir checkpoints
#    cd checkpoints
#    azcopy copy "$blob_dir/checkpoints/Auto-UI-Base$SAS" ./ --recursive
#    azcopy copy "$blob_dir/checkpoints/Qwen2-VL-7B-Instruct$SAS" ./ --recursive
#    azcopy copy "$blob_dir/checkpoints/blip2-opt-2.7b$SAS" ./ --recursive
#    azcopy copy "$blob_dir/checkpoints/roberta-base$SAS" ./ --recursive
#    cd ../
#fi
#
#if [ ! -d "images" ]; then
#    azcopy copy "$blob_dir/images$SAS" ./ --recursive
#fi


#conda create -n digirl python==3.10 -y
#conda activate digirl
#pip install -r digirl_requirements.txt
#sudo apt-get update
#sudo apt-get install libglib2.0-0 -y
#pip install openai azure-identity-broker --upgrade
#
#git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
#cd LLaMA-Factory
#pip install -e ".[torch,metrics]"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python /cosmos/zhengjiani/DPO/DPO_train.py
