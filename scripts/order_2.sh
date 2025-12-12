#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/dengkn/.cache/huggingface
export LC_ALL=zh_CN.UTF-8
port=$(shuf -i25000-30000 -n1)

# bash scripts/order_2.sh> logs_and_outputs/order_2/logs/train_and_infer.log 2>&1 &

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/$1/1-dbpedia \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round1 \
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
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 1 \
   --galore_update_proj_gap 10 \
   --lamda_3 0.05

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_2/$1/1-dbpedia/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/$1/2-amazon \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round2 \
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
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 1 \
   --galore_update_proj_gap 10 \
   --lamda_3 0.05

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_2/$1/2-amazon/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/$1/3-agnews \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round3 \
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
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 1 \
   --galore_update_proj_gap 10 \
   --lamda_3 0.05

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_2/$1/3-agnews/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_2/$1/4-yahoo \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 2 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order2_round4 \
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
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 1 \
   --galore_update_proj_gap 10 \
   --lamda_3 0.05
