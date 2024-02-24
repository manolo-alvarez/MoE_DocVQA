#!/bin/bash

moe_mode="sparse"
num_experts=2
top_k_experts=1
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="data/json"
IMAGE_FOLDER="data/img"
cd ~/MoE-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 deepspeed moellava/train/train_xformers.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ./utils/zero2.json \
    --model_name_or_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384 \
    --version phi \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./experiments/SFT/checkpoints/llavaphi-2.7b-finetune-moe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
