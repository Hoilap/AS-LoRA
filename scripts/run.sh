bash scripts/order_1.sh outputs_order_1 8 "localhost:0,1,2,3,4,5,6,7" 1e-05 ".*EncDecAttention.(q|v).*" ".*SelfAttention.(q|v).*" ".*SelfAttention.(q|v).loranew_A.*" > logs/order_1.log 2>&1
sleep 5
bash scripts/order_2.sh outputs_order_2 8 "localhost:0,1,2,3,4,5,6,7" 1e-05 ".*EncDecAttention.(q|v).*" ".*SelfAttention.(q|v).*" ".*SelfAttention.(q|v).loranew_A.*" > logs/order_2.log 2>&1
sleep 5
bash scripts/order_3.sh outputs_order_3 8 "localhost:0,1,2,3,4,5,6,7" 1e-05 ".*EncDecAttention.(q|v).*" ".*SelfAttention.(q|v).*" ".*SelfAttention.(q|v).loranew_A.*" > logs/order_3.log 2>&1
sleep 5
bash scripts/order_4.sh 1e-03 32 "localhost:0,1,2,3,4,5,6,7" 1e-05 ".*EncDecAttention.(q|v).*" ".*SelfAttention.(q|v).*" ".*SelfAttention.(q|v).loranew_A.*" outputs_order_4 > logs/order_4.log 2>&1
sleep 5
bash scripts/order_5.sh 1e-03 32 "localhost:0,1,2,3,4,5,6,7" 1e-05 ".*EncDecAttention.(q|v).*" ".*SelfAttention.(q|v).*" ".*SelfAttention.(q|v).loranew_A.*" outputs_order_5 > logs/order_5.log 2>&1
sleep 5
bash scripts/order_6.sh 1e-03 32 "localhost:0,1,2,3,4,5,6,7" 1e-05 ".*EncDecAttention.(q|v).*" ".*SelfAttention.(q|v).*" ".*SelfAttention.(q|v).loranew_A.*" outputs_order_6 > logs/order_6.log 2>&1
sleep 5