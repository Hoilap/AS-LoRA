#!/bin/bash

bash scripts_llama/order_1.sh outputs_order_1 2 2e-04 ".*mlp.gate_proj.*" "localhost:0,1" 1e-06 > logs_llama/order_1.log 2>&1
sleep 5
bash scripts_llama/order_2.sh outputs_order_2 2 2e-04 ".*mlp.gate_proj.*" "localhost:0,1" 1e-06 > logs_llama/order_2.log 2>&1
sleep 5
bash scripts_llama/order_3.sh outputs_order_3 2 2e-04 ".*mlp.gate_proj.*" "localhost:0,1" 1e-06 > logs_llama/order_3.log 2>&1
