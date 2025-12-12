#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/home/dengkn/.cache/huggingface
export LC_ALL=zh_CN.UTF-8
port=$(shuf -i25000-30000 -n1)
 
deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/MNLI \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/1-MNLI \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round1 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0
   

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/1-MNLI/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/CB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/2-CB \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round2 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/2-CB/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/WiC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/3-WiC \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round3 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/3-WiC/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/COPA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/4-COPA \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round4 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0 

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/4-COPA/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/QQP \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/5-QQP \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round5 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0  

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/5-QQP/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/BoolQA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/6-BoolQA \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round6 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/6-BoolQA/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/RTE \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/7-RTE \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round7 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0 

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/7-RTE/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/IMDB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/8-IMDB \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round8 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/8-IMDB/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/9-yelp \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round9 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/9-yelp/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/10-amazon \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round10 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/10-amazon/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/SST-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/11-SST-2 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round11 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/11-SST-2/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/12-dbpedia \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round12 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/12-dbpedia/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/13-agnews \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round13 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/13-agnews/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/MultiRC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/14-MultiRC \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round14 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0

sleep 5

deepspeed --include $3 --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/order_4/$8/14-MultiRC/t5_adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/order_4/$8/15-yahoo \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate $1 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round15 \
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
   --lamda_2 0.1 \
   --galore_rank $2 \
   --galore_lr $4 \
   --lora_modules $6 \
   --optim_target_modules $5 \
   --proj_lora_modules $7 \
   --galore_scale 0.25 \
   --lamda_3 0