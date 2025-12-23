#!/bin/bash

# 低显存版本：使用ZeRO-2 + CPU offload
bash scripts_llama/order_1_optimized_lowmem_test.sh outputs_order_1_lowmem 2 2e-04 ".*mlp.gate_proj.*" "localhost:0,1" 1e-06 > logs_llama/order_1_optimized_lowmem.log 2>&1
