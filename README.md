# Continual Gradient Low-Rank Projection Fine-Tuning for LLMs
- Official Code for Continual Gradient Low-Rank Projection Fine-Tuning for LLMs
- It is built based on the pretrained T5-large model and llama2 model, and finetuned on our data.

## FrameWork
![image_text](framework/framework.jpg)


## GORP 项目文件结构详解

这是一个基于**持续学习(Continual Learning)**的LLM微调项目,使用**Gradient Low-Rank Projection (GORP)** 方法,结合**LoRA**技术进行参数高效微调。

### 📁 **一、顶层文件**

- **README.md**: 项目说明文档,介绍如何安装、训练和评估模型
- **requirements.txt**: Python依赖包列表(transformers, deepspeed, peft等)

### 📁 **二、CL_Benchmark/** - 持续学习基准数据集

包含多个NLP任务的数据集,用于测试模型的持续学习能力:

- **`BoolQA/`**: 布尔问答任务
- **`COPA/`**: 因果推理任务
- **`MultiRC/`**: 多项选择阅读理解
- **`NLI/`**: 自然语言推理(CB, MNLI, RTE子任务)
- **`QQP/`**: 问题对相似度判断
- **`SC/`**: 情感分类(amazon, IMDB, SST-2, yelp)
- **`TC/`**: 文本分类(agnews, dbpedia, yahoo)
- **`WiC/`**: 词义消歧

### 📁 **三、configs/** - 配置文件目录

#### **instruction_config.json**
- 每个任务的指令模板(zero-shot prompts)

#### **`ds_configs/`** - DeepSpeed配置
- `stage0/1/2/3.config`: 不同阶段的分布式训练配置
- `eval.config`: 评估配置
- `stage2_llama.config`: LLaMA模型专用配置

#### **`order1~6_configs/`** - 任务顺序配置
- 定义6种不同的任务学习顺序
- 每个文件夹包含各任务的`train/dev/test_tasks.json`
- 在持续学习的设定中，模型需要按照特定的时间顺序依次学习多个不同的任务。由于模型容易出现“灾难性遗忘”（即学会了新任务忘了旧任务），学习任务的先后顺序往往会对最终的性能产生很大影响。不同的任务顺序配置展现出鲁棒性
---

### 📁 **四、data/** - 数据处理

- **`generate_labels.py`**: 生成标签数据的脚本
- **`gpt2tokenizer/`**: GPT-2分词器配置文件

### 📁 **五、src/** - 核心源代码 ⭐⭐⭐


#### 主训练文件
- **run_uie_lora.py** (628行):   UIE (Lu et al., ACL 2022) 是一个非常有名的基于 T5 模型（Text-to-Text Transfer Transformer）的框架。
	- **主入口文件**,负责训练流程控制
	-  环境与日志设置 (main 开头部分)
		- HfArgumentParser: 解析命令行参数或 JSON 配置文件。
		- logging: 设置日志级别，确保在分布式训练中只有主进程（Rank 0）输出详细日志。
		- set_seed: 固定随机种子，保证实验可复现。
	- 数据集加载
		- load_dataset(..., "uie_dataset_lora.py"): 加载自定义的 UIE 数据集。
		- 这暗示任务是将“实体抽取”、“关系抽取”等任务转化为生成式（Seq2Seq 或 Causal LM）的形式。
	- 模型与分词器加载 (核心逻辑)
		- 这段逻辑处理了 LLaMA 和 Encoder-Decoder (如 T5) 两类模型：
		- 分词器 (Tokenizer)：针对 LLaMA 特殊处理了 bos, eos, pad token id。
		- 模型架构选择：
			- LLaMA 使用 LlamaForCausalLM_with_lossmask（这是一个自定义类，通常用于在计算 Loss 时屏蔽掉 Prompt 部分，只计算回答部分的 Loss）。
			- 其他模型使用 AutoModelForSeq2SeqLM。
		- 加载 LoRA：
			- 情况1（增量训练）：如果路径中包含 adapter，说明是加载一个已经训练过的 LoRA 权重 (PeftModel.from_pretrained)。
			- 情况2（从头微调）：如果是新训练，则初始化一个新的 LoraConfig 并用 get_peft_model 包装模型。
	- 参数冻结与解冻策略 (精细控制)
		-  它区分了 loranew_（当前要训练的）和 lora_（之前训练好但现在要固定的）。这通常需要修改 peft 源码或手动重命名参数层来实现。
	- 数据整理器 (DataCollatorForUIE)
		- 负责将文本数据 batch 化，处理 padding (longest)。
		- 处理 LLaMA 和 T5 不同的 padding 逻辑。
	- **自定义 Trainer (UIETrainer)**
		- 实例化了 UIETrainer。结合之前的 lamda 参数，这个 Trainer 内部肯定重写了 compute_loss 方法，**利用 lamda 参数来实现你想要的正则化逻辑**。
		- 如果开启 denser_evaluation，会加入额外的回调函数。
	- 训练后处理
		- 保存模型：只保存 Adapter 部分 (peft_model_id)，不保存底座模型，节省空间。
		- 梯度矩阵分析：grad_mat = trainer.get_grad_matrix_from_grad()

| 场景 | 输入示例 (model_name_or_path) | 触发逻辑 | 行为总结 | 最终模型状态 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 加载已训好的 Adapter** | `xxx/t5-adapter` <br> `xxx/llama-adapter` | 路径含 `"adapter"` | **先读配置找底座，再挂载权重**。<br>1. 读取 Adapter 配置文件，找到底座模型（如 `t5-large`）。<br>2. 加载底座模型。 <br>3. 使用 `PeftModel` 将 Adapter 权重合并进去。 | **包含旧权重的 PEFT 模型**<br>(用于增量训练或推理) |
| **2. LLaMA 从头微调** | `meta-llama/Llama-2-7b` | 路径含 `"llama"` | **特化加载 + 初始化 LoRA**。<br>1. 使用 `LlamaTokenizer` 并手动修正特殊 Token ID。<br>2. 加载 **魔改版** 模型类 `LlamaForCausalLM_with_lossmask`。<br>3. 初始化全新的 LoRA 层。 | **初始化的 PEFT 模型**<br>(Task: Causal LM) |
| **3. T5/BART 从头微调** | `t5-large` <br> `facebook/bart-large` | 既无 `"adapter"`<br>也无 `"llama"` | **通用加载 + 初始化 LoRA**。<br>1. 使用标准的 `AutoTokenizer` 和 `AutoModelForSeq2SeqLM`。<br>2. 初始化全新的 LoRA 层。 | **初始化的 PEFT 模型**<br>(Task: Seq2Seq) |

#### **训练器**
- **uie_trainer_lora.py** (1297行):
  - 自定义Seq2Seq训练器
  - 实现梯度更新、评估回调等
  - **training_step：里面包含了训练过程（前向传播+计算损失）**

#### **数据处理**
- **uie_dataset_lora.py** (519行):
  - 数据集加载和预处理
  - 支持多任务、多指令策略
- **uie_collator.py** (227行):
  - 数据批处理collator
  - 支持encoder-decoder和decoder-only模型

#### **评估**
- **`compute_metrics.py`**: 计算评估指标
- **evaluator.py** (906行): 详细的评估器实现(准确率、F1等)

---

### 📁 **六、src/Proj_torch/** - 核心算法实现 🔥🔥🔥

**这是修改算法核心代码的地方!**

#### **proj_projector.py** (105行)
- **`ProjProjector`类**: 实现梯度低秩投影的核心算法
- **关键方法**:
  - `project()`: 将全秩梯度投影到低秩子空间
  - `project_back()`: 将低秩梯度投影回原空间
  - `get_orthogonal_matrix()`: 通过SVD分解获取正交矩阵
- **投影类型**: 支持std、reverse_std、left、right、full等多种投影方式

#### **proj_projector_tensor.py**
- **`ProjProjectorTensor`类**: 用于处理高维张量的投影(>2维)

#### **adamw.py** (186行)
- **`AdamW`优化器**: 自定义的AdamW实现
- **核心功能**:
  - 在`step()`方法中集成梯度投影
  - 动态更新投影矩阵(每`update_proj_gap`步)
  - 支持梯度投影前后的处理

#### **`__init__.py`**
- 导出`ProjAdamW`优化器供外部使用


### 📁 **七、src/peft/** - LoRA实现

修改自HuggingFace PEFT库:

- **lora.py** (748行):
  - **LoRA实现**: 定义LoRA层和配置【里面就定义了loranew参数】
  - **关键修改**: 
    - `r_sum`: 记录之前LoRA参数维度(用于持续学习)
    - `save_loranew`: 是否独立保存新LoRA参数
- **`peft_model.py`**: PEFT模型封装
- **`mapping.py`**: 模型映射
- **其他tuners**: adalora, prefix_tuning等

### 📁 **八、src/model/** - 模型定义
- **`llama.py`**: LLaMA模型的特殊处理

### 📁 **九、src/rouge/** - 评估指标
- ROUGE分数计算(用于文本生成任务评估)


### 📁 十、scripts/ - 训练脚本，scripts_llama/ - LLaMA专用脚本
- **run.sh**: 主运行脚本,依次执行order_1到order_6
- **`order_1.sh ~ order_6.sh`**: 
  - 每个脚本定义一种任务学习顺序
  - 包含多个任务的连续训练
  - 配置学习率、批大小、LoRA模块等
- **`run_llama.sh`**: LLaMA模型的训练流程
- **`order_1/2/3.sh`**: LLaMA的三种任务顺序

### 📁 **十二、logs/ 和 logs_llama/** - 日志目录
- 存储训练和推理日志
- `null.txt`: 占位符文件


## 💡 **算法工作流程**

1. **数据加载**: uie_dataset_lora.py → 加载任务数据
2. **模型初始化**: run_uie_lora.py → 加载T5/LLaMA + LoRA层
3. **训练循环**: uie_trainer_lora.py → 控制训练过程
4. **梯度计算**: 反向传播得到全秩梯度
5. **梯度投影**: `ProjProjector.project()` → 降低梯度维度
6. **优化更新**: `AdamW.step()` → 在低秩空间更新参数
7. **投影回原空间**: `project_back()` → 应用到原模型参数
8. **持续学习**: 保存旧LoRA参数,为新任务添加新LoRA层






## Question
### 参数命名
第一阶段 LoRA (旧参数) - 来自上一个任务
"encoder.block.0.layer.0.SelfAttention.q.lora_A.weight"       # r × in_features
"encoder.block.0.layer.0.SelfAttention.q.lora_B.weight"       # out_features × r

第二阶段 LoRA (新参数) - 当前任务 【你添加的】
"encoder.block.0.layer.0.SelfAttention.q.loranew_A.weight"    # r_new × in_features
"encoder.block.0.layer.0.SelfAttention.q.loranew_B.weight"    # out_features × r_new

### 搞清楚训练时训练的是哪一个参数？
```
for name, param in model.named_parameters():
    if name.find("loranew_") != -1:
        param.requires_grad = True          # ✅ loranew_A/B 可以训练
    elif name.find("lora_") != -1:
        param.requires_grad = False         # ❌ lora_A/B 被冻结
    # 其他参数的处理...
```
> fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
为什么这样搞？工程可能在做持续学习（Continual Learning）或多阶段微调

#### 情况1️⃣：当 model_name_or_path 是**本地路径**

```
参数: --model_name_or_path initial_model/llama
     |
     v
检查 initial_model/llama 是否存在？
     |
     ├─ YES ✅
     |   └─ 直接从本地加载
     |       ├─ 读取 config.json
     |       ├─ 读取 pytorch_model.bin
     |       ├─ 读取 tokenizer.model
     |       └─ 返回模型对象
     |
     └─ NO ❌
         └─ 报错：FileNotFoundError
             (因为代码认为你给的是本地路径，不会尝试网络下载)
```

**关键点**：如果你提供的是本地路径，HuggingFace 库**不会自动从网络下载**！

#### 情况2️⃣：当 model_name_or_path 是 **HuggingFace model ID**

```
参数: --model_name_or_path meta-llama/Llama-2-7b
     |
     v
from_pretrained() 的加载优先级:

优先级1️⃣: 检查是否是本地路径？
    └─ NO (因为这是 model ID 格式)

优先级2️⃣: 检查 cache_dir 缓存目录
    默认缓存目录: ~/.cache/huggingface/hub/
    ├─ 存在完整模型？
    |   ├─ YES ✅ 从缓存加载
    |   └─ NO 继续
    └─ 存在不完整下载？
        └─ YES 继续断点下载

优先级3️⃣: 从网络下载
    ├─ 连接 HuggingFace 官方服务器
    ├─ 下载模型文件到缓存目录
    ├─ 解压/验证
    └─ 加载到内存
```
- **代码中没有自动下载模型的功能** - 脚本中的 `from_pretrained()` 方法可以从 HuggingFace 自动下载模型，但需要网络连接且速度可能较慢- **代码中有**：`src/run_uie_lora.py` 使用 HuggingFace 的 `from_pretrained()` 方法
- 这个方法可以自动从 HuggingFace 下载模型（如果传入 model ID 而不是本地路径）
- **原作者硬编码了模型路径** - 使用的是 `/data/chenxu/models/t5-large` 和 `/data/chenxu/models/llama`，这是他的本地路径   以下脚本文件已全部更新，将模型路径从 `/data/chenxu/models/` 改为 `initial_model/`：

### 内存计算/OOM问题踩坑、
```
当前配置：
  模型: LLaMA-7B (约13GB fp32)
  LoRA rank: 8
  训练方式: DeepSpeed ZeRO Stage 2 + LoRA + GaLore
  Batch size: 1 per device
  Gradient accumulation: 8
  Gradient checkpointing: 启用
  精度: bfloat16
  GPU: 2 × 24GB (并行训练)
内存消耗分解：
  模型权重 (bfloat16): ~13GB
  LoRA 参数: 很小 (rank=8, 约100MB)
  优化器状态 (ZeRO-2分片): ~6-8GB
  梯度: ~6-7GB (ZeRO-2分片)
  激活值 (即使有gradient checkpointing): ~2-4GB
  临时缓冲区: ~1-2GB
  单卡总需求: 约 23-24GB
RTX 4090 显存: 24GB
```
### try:用了34364MiB=33GB, fp32,时间：1642-2009，约3个小时(one task)====重复不出来


### 不兼容报错
```
conda activate aslora && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'numpy<2' 'pyarrow==10.0.1' 'datasets==2.13.1' 'fsspec==2023.6.0' 'tqdm==4.65.0'
```
### 尝试zero3失败
无论 ZeRO-3 是否 offload，只要启用都会出现隐藏维度为 0 的错误。这说明问题根本不在 offload，而在 ZeRO-3 与 gradient_checkpointing 的深层兼容性问题，或者与 PEFT/LoRA 在分片下的参数重构问题。


### 脚本阅读
```
bash scripts_llama/order_1_optimized.sh outputs_order_1 2 2e-04 ".*mlp.gate_proj.*" "localhost:0,1" 1e-06
#                                         $1           $2  $3       $4                $5              $6


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
   --optim_target_modules $4 \  //说明Galore被启用了
   --proj_lora_modules ".*self_attn.(q_proj|v_proj).loranew_A.*" \
   --galore_rank $2 \  
   --galore_scale 0.25 \
   --galore_lr $6 \
   --gradient_checkpointing True
```
打印出来`trainable params: 1447034880 || all params: 6742609920 || trainable%: 21.461049907511185`，说明galore全参训练了

`12/15/2025 13:37:48 - WARNING - __main__ - Process rank: 1, device: cuda:1, n_gpu: 1distributed training: True, 16-bits training: False` 说明使用了32位
```
if 'adapter' in model_args.model_name_or_path: # add lora-adapter to the original model
        model = model_class.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if training_args.fp16 else torch.float32  #新加，权重的加载方式
        )  #加载 底座模型 的权重 
```
在代码中做出如下修正，解决不知道导入精度为多少的问题

### GORP用不了BF16
```
RuntimeError: cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED
File ".../Proj_torch/proj_projector.py", line 83, in get_orthogonal_matrix
    U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
```
原因：SVD (奇异值分解) 是一个数值计算密集的操作，对数值精度要求极高。当用 float16 进行 SVD 时，会出现数值不稳定的问题。


### 报错：配置精度出现冲突
```
ValueError: Please correct the following DeepSpeed config values that mismatch TrainingArguments values:
- ds fp16.enabled=True vs hf fp16|fp16_full_eval+fp16_backend(amp)=False
```
DeepSpeed 配置文件 ([stage2_llama.config](configs/ds_configs/stage2_llama.config)stage2_llama.config)：
设置了 fp16.enabled=True（启用半精度浮点）
HuggingFace 训练参数：
没有启用 fp16（默认使用 float32）
即你的脚本中没有 --fp16 参数
基座模型加载时只看命令行参数（--bf16/--fp16），不看 DeepSpeed 配置文件。
阶段2：DeepSpeed 初始化（Trainer 创建时）

DeepSpeed 读取配置文件（如 stage2_llama.config）
如果配置与模型加载精度不一致，DeepSpeed 会尝试转换
但转换有额外开销和风险
你的 DeepSpeed 配置：
{
    "bfloat16": {"enabled": true},
    "fp16": {"enabled": "auto"}
}
bf16: true → DeepSpeed 期望 BF16
fp16: auto → 根据命令行自动决定

### 什么是CPU offload

标准方式：
模型参数 → GPU → 计算 → 反向传播

CPU Offload：
模型参数 → CPU (存储) 
计算时 → 把参数移到GPU → 计算 → 移回CPU
情况1：使用了 Stage 2 + CPU Offload（可能）
从你的脚本：
```
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```


dbpedia 43151MiB
amazon 17562MiB



- try: order_1_optimized.sh  不能跑：
```
============================================================
12/15/2025 15:50:55 - WARNING - __main__ - 模型参数精度验证:
12/15/2025 15:50:55 - WARNING - __main__ - ============================================================
12/15/2025 15:50:55 - WARNING - __main__ -   float16: 291 个参数
12/15/2025 15:50:55 - WARNING - __main__ -     示例参数: ['base_model.model.model.embed_tokens.weight', 'base_model.model.model.layers.0.self_attn.q_proj.weight', 'base_model.model.model.layers.0.self_attn.k_proj.weight']
12/15/2025 15:50:55 - WARNING - __main__ -   float32: 256 个参数
12/15/2025 15:50:55 - WARNING - __main__ -     示例参数: ['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight', 'base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight']
12/15/2025 15:50:55 - WARNING - __main__ - ============================================================

12/15/2025 15:50:55 - WARNING - __main__ - ✓ 第一个参数精度: torch.float16
12/15/2025 15:50:55 - WARNING - __main__ - ✓ 第一个参数设备: cpu

-----Gradient checkpointing: True -----
trainable params: 1447034880 || all params: 6742609920 || trainable%: 21.461049907511185
```
- try：能跑？第10个样例卡住
```
-----Gradient checkpointing: True -----
12/15/2025 16:21:18 - WARNING - __main__ - 
============================================================
12/15/2025 16:21:18 - WARNING - __main__ - 模型参数精度验证:
12/15/2025 16:21:18 - WARNING - __main__ - ============================================================
12/15/2025 16:21:18 - WARNING - __main__ -   float16: 291 个参数
12/15/2025 16:21:18 - WARNING - __main__ -     示例参数: ['base_model.model.model.embed_tokens.weight', 'base_model.model.model.layers.0.self_attn.q_proj.weight', 'base_model.model.model.layers.0.self_attn.k_proj.weight']
12/15/2025 16:21:18 - WARNING - __main__ -   float32: 256 个参数
12/15/2025 16:21:18 - WARNING - __main__ -     示例参数: ['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight', 'base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight']
12/15/2025 16:21:18 - WARNING - __main__ - ============================================================

12/15/2025 16:21:18 - WARNING - __main__ - ✓ 第一个参数精度: torch.float16
12/15/2025 16:21:18 - WARNING - __main__ - ✓ 第一个参数设备: cpu
```


### 过程中出现脚本一样但一个OOM另一个没有的问题？  **未解决**
为什么不一样？order_1有检查点而order_1_optimized_lowmem_test没有
```
resume_from_checkpoint (`str` or `bool`, *optional*):
    If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
    `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
    of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
```
但根据代码逻辑 line 304，如果 output_dir 已存在且有 checkpoint，会自动 resume。为了完全避免任何 resume 行为，建议显式添加：
这样可以明确禁止任何 checkpoint 加载。




### deepspeed_init
1. deepspeed_init 会不会自动把 self.model 变成 Engine？
答案：是的，它应该会（有副作用）。
deepspeed_init 是 Hugging Face Trainer 内部的一个辅助函数。它的作用流程通常如下：
它读取 training_args.deepspeed 配置文件。
它调用底层的 deepspeed.initialize(model=self.model, ...)。
关键点：它会把初始化好的 DeepSpeed Engine 赋值回 self.model，同时也会更新 self.optimizer 和 self.lr_scheduler。
它通常还会把 self.deepspeed 这个属性指向这个 Engine。
但是，在你之前的报错中：
AttributeError: 'LlamaForCausalLM_with_lossmask' object has no attribute 'backward'
这说明虽然 deepspeed_init 运行了，但在那个特定的时间点，代码引用的对象（self.deepspeed）并没有成功指向 Engine，或者 self.model 被某些后续操作（比如 _wrap_model）又变回了原始模型。
2. DeepSpeed 是 Accelerator 吗？
不完全是，它们是两层关系。
DeepSpeed: 是一个优化引擎（Engine）。它直接接管模型的训练、梯度更新、显存管理（ZeRO）。它有点像是一个“超级优化版的 PyTorch”。
Accelerator (Hugging Face accelerate 库): 是一个**“统筹者”或“包装器”**。它的作用是让同一套代码可以运行在 GPU、TPU、多卡、DeepSpeed 等不同环境上。
在你的代码中：
self.accelerator 是 accelerate 库的对象。当你在参数里开启 DeepSpeed 时，accelerator 内部 会集成 DeepSpeed。
通常的写法有两种：
旧写法（你的代码偏向这种）：手动调用 deepspeed_init，显式管理 DeepSpeed。
新写法（推荐）：完全不调 deepspeed_init，直接用 model, optim = accelerator.prepare(model, optim)。accelerator 会自动发现你要用 DeepSpeed 并帮你初始化好一切。

### try:用了26736MiB， fp32, 删除Galore微调（否则OOM）   失败，出现分布式通信的问题

在 Hugging Face 的 TrainingArguments 中将 fp16 设置为 True，开启的就是“混合精度训练”（Mixed Precision Training），而不是“纯FP16训练”。
它的内部工作机制正如你所描述的：“计算用 16位，更新用 32位”。





[rank1]: ValueError: Argument `synced_gpus` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/dengkn/N-LoRA/src/run_N_lora.py", line 620, in <module>
[rank0]:     main()
[rank0]:   File "/home/dengkn/N-LoRA/src/run_N_lora.py", line 597, in main
[rank0]:     predict_results = trainer.predict(
[rank0]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/trainer_seq2seq.py", line 244, in predict
[rank0]:     return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
[rank0]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/trainer.py", line 3754, in predict
[rank0]:     output = eval_loop(
[rank0]:   File "/home/dengkn/N-LoRA/src/uie_trainer_lora.py", line 203, in evaluation_loop
[rank0]:     loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
[rank0]:   File "/home/dengkn/N-LoRA/src/uie_trainer_lora.py", line 337, in prediction_step
[rank0]:     generation_config = GenerationConfig(**gen_kwargs)
[rank0]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/generation/configuration_utils.py", line 453, in __init__
[rank0]:     self.validate(is_init=True)
[rank0]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/generation/configuration_utils.py", line 725, in validate
[rank0]:     raise ValueError(
[rank0]: ValueError: Argument `synced_gpus` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.

## Setup

You can install the required libraries by running 

```
pip install -r requirements.txt
```

You are also required to download the t5-large model from huggingface, put it to the folder named ```initial_model```, and rename the model folder as 't5-large'.

LLaMA2 HF is also supported. You can put your llama2 hf model to the folder named ```initial_model``` and rename the model folder as 'llama'.


## Training and Evaluation

For t5-large:

You can reproduce our experiments of order 1 to 6 by simply running ```scripts/run.sh```.

The model you have trained will be saved in ```logs_and_outputs/order_(1 to 6)/outputs_order_(1 to 6)```.

The result of each task will be saved in ```logs_and_outputs/order_(1 to 6)/outputs/TASK_NAME/predict_results.json```.

You can also check the logs during training and infering in  ```logs/order_(1 to 6).log```

For LLaMA2:

You can reproduce our experiments of order 1 to 3 by simply running ```scripts_llama/run_llama.sh```.

The model you have trained will be saved in ```logs_and_outputs_llama/order_1(2 or 3)/outputs```.

The result of each task will be saved in ```logs_and_outputs_llama/order_1(2 or 3)/outputs/TASK_NAME/predict_results.json```.

You can also check the logs during training and infering in  ```logs_llama/order_1(2 or 3)/order_1(2 or 3).log```


## Citation
```markdown
@inproceedings{wang-etal-2025-continual,
    title = "Continual Gradient Low-Rank Projection Fine-Tuning for {LLM}s",
    author = "Wang, Chenxu  and
      Lyu, Yilin  and
      Sun, Zicheng  and
      Jing, Liping",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    pages = "14815--14829",
}
```
## Acknowledgment
We acknowledge the publicly available codebase of [O-LoRA](https://github.com/cmnfriend/O-LoRA)
