#!/bin/bash

bash scripts_llama/order_1_optimized.sh outputs_order_1 2 2e-04 ".*mlp.gate_proj.*" "localhost:0,1" 1e-06 > logs_llama/order_1.log 2>&1
