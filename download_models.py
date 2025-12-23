#!/usr/bin/env python3
"""
下载模型到 initial_model 目录
"""
import os
from transformers import AutoModel, AutoTokenizer

# 创建 initial_model 目录
os.makedirs("initial_model", exist_ok=True)

# 在脚本内部设置 Hugging Face 镜像端点（若环境已设置则不覆盖）
HF_MIRROR = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_MIRROR
print(f"已设置 HF_ENDPOINT = {HF_MIRROR}")

print("开始下载 T5-large 模型...")
# 下载 T5-large
t5_model = AutoModel.from_pretrained("google-t5/t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")

# 保存到 initial_model/t5-large
t5_model.save_pretrained("initial_model/t5-large")
t5_tokenizer.save_pretrained("initial_model/t5-large")
print("✓ T5-large 模型已保存到 initial_model/t5-large")

# 使用 ModelScope 下载 LLaMA2 模型
# 注意：取消注释以下内容来下载 LLaMA2
try:
    from modelscope import snapshot_download
    print("开始下载 LLaMA2 模型...")
    #model_dir = snapshot_download('LLM-Research/llama-2-7b',local_dir='initial_model/llama')
    model_dir = snapshot_download('shakechen/Llama-2-7b-chat-hf',local_dir='initial_model/llama')
    print("✓ LLaMA 模型已保存到 initial_model/llama")
except ImportError:
    print("⚠ 若要下载 LLaMA2，请先安装 modelscope: pip install modelscope")
except Exception as e:
    print(f"❌ LLaMA 下载失败: {str(e)}")

print("\n下载完成！")
print("模型目录结构:")
os.system("ls -la initial_model/")
