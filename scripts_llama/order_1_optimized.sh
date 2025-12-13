#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/dengkn/.cache/huggingface
export LC_ALL=zh_CN.UTF-8
# 启用更优的 PyTorch CUDA 显存分配，可节省 5-10% 显存
export PYTORCH_ALLOC_CONF=expandable_segments:True
port=$(shuf -i25000-30000 -n1)
 
# bash scripts_llama/order_1.sh> logs_and_outputs_llama/order_1/logs/train_and_infer.log 2>&1 &

deepspeed --include $5 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/llama \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/order_1/$1/1-dbpedia \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --learning_rate $3 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --lora_modules ".*self_attn.(q_proj|v_proj).*" \
   --optim_target_modules $4 \
   --proj_lora_modules ".*self_attn.(q_proj|v_proj).loranew_A.*" \
   --galore_rank $2 \
   --galore_scale 0.25 \
   --galore_lr $6 \
   --gradient_checkpointing True

