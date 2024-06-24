#! /bin/bash
export CUDA_VISIBLE_DEVICES='0'
export WANDB_PROJECT=
export WANDB_RUN_ID=
export WANDB_RESUME=allow
model_name_or_path=

train_file=
validation_file=
output_dir="/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

mkdir -p ${cache_dir}
cutoff_len=512

torchrun --nproc_per_node 4 --master_port 29500 lora_ftg.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --output_dir ${output_dir} \