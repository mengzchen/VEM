#!/usr/bin/env bash

# i1 -A berzelius-2024-341 --gres=gpu:1  --time 12:00:00 --mail-type "BEGIN,END,FAIL" --mail-user "v-zhengjiani@microsoft.com"
# squeue -u x_wenyi -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
# export PATH=/home/x_wenyi/.conda/envs/digirl/bin:$PATH

# python models/cogagent_demo.py
# python3 eval_aitw.py --config configs/online_eval_cogagent.yaml
# python3 eval_aitw.py --config configs/online_eval_autogui.yaml
# python3 eval_aitw.py --config configs/online_eval_our.yaml
# python3 eval_aitw.py --config configs/online_eval_digirl.yaml