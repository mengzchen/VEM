#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

save_name=debug             
micro_batch_size=16
use_ckpt=False
train_epochs=10
learning_rate=1e-5
nproc_per_node=1
data_path=data/mind2web_train_sft.json
qwen_ckpt=checkpoints/Qwen2-VL-7B-Instruct
pretrain_ckpt=checkpoints/Qwen2-VL-7B-Instruct
save_path=checkpoints/debug

GPUS_PER_NODE=${nproc_per_node}
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL=${pretrain_ckpt}
QWEN_PATH=${qwen_ckpt}
DATA=${data_path}
SAVE_PATH="${save_path}/${SAVE_NAME}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --qwen_path $QWEN_PATH \
    --data_path $DATA \
    --bf16 True \
    --fix_vit False \
    --output_dir $SAVE_PATH \
    --num_train_epochs ${train_epochs} \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 30 \
    --learning_rate ${learning_rate} \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --model_max_length 768 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed configs/deepspeed_zero2.json