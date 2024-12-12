#!/usr/bin/env bash

# export PATH=/home/x_wenyi/.conda/envs/digirl/bin:$PATH
CUDA_VISIBLE_DEVICES=0 python3 train_rl.py --config-path=configs/ --config-name=digirl_online