#!/bin/bash
set -x

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/dengkn/.cache/huggingface
export LC_ALL=en_US.UTF-8  # 改为en_US避免locale警告
# 启用更优的 PyTorch CUDA 显存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# 减少碎片，解决了第一个任务总是OOM的问题
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

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
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 16 \
   --learning_rate $3 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order1_round1 \
   --max_source_length 384 \
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
   --proj_lora_modules ".*self_attn.(q_proj|v_proj).loranew_A.*" \
   --galore_rank $2 \
   --galore_scale 0.25 \
   --galore_lr $6 \
   --gradient_checkpointing True\

  

sleep 5

deepspeed --include $5 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/order_1/$1/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/order_1/$1/2-amazon \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 8 \
   --learning_rate $3 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order1_round2 \
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
   --proj_lora_modules ".*self_attn.(q_proj|v_proj).loranew_A.*" \
   --galore_rank $2 \
   --galore_scale 0.25 \
   --galore_lr $6 \
   --gradient_checkpointing True\


sleep 5

deepspeed --include $5 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/order_1/$1/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/order_1/$1/3-yahoo \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 8 \
   --learning_rate $3 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order1_round3 \
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
   --proj_lora_modules ".*self_attn.(q_proj|v_proj).loranew_A.*" \
   --galore_rank $2 \
   --galore_scale 0.25 \
   --galore_lr $6 \
   --gradient_checkpointing True\



sleep 5

deepspeed --include $5 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/order_1/$1/3-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/order_1/$1/4-agnews \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 8 \
   --learning_rate $3 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order1_round4 \
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
   --proj_lora_modules ".*self_attn.(q_proj|v_proj).loranew_A.*" \
   --galore_rank $2 \
   --galore_scale 0.25 \
   --galore_lr $6 \
   --gradient_checkpointing True\

