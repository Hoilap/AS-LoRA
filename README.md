# Continual Gradient Low-Rank Projection Fine-Tuning for LLMs
- Official Code for Continual Gradient Low-Rank Projection Fine-Tuning for LLMs
- It is built based on the pretrained T5-large model and llama2 model, and finetuned on our data.

## FrameWork
![image_text](framework/framework.jpg)


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

You can reproduce our experiments of order 1 to 3 by simply running ```scripts/run_llama.sh```.

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
