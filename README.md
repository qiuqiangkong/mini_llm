# Minimal PyTorch implementation of Llama

This repository provides a minimal PyTorch implementation of LLaMA. It simplifies the LLaMA code [1] for easier understanding and usage. The model trains a natural language generation system on 1 million characters and converges in under 5 minutes.

## 0. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/mini_llm
cd mini_llm

# Install Python environment
conda create --name mini_llm python=3.10

# Activate environment
conda activate mini_llm

# Install Python packages dependencies
bash env.sh
```

## 1. Train

```python
CUDA_VISIBLE_DEVICES=0 python train.py --model_name=Llama
```

We train the language model on the Shakespeare dataset with 1 million characters. Training takes around 20 minutes for 10,000 steps on a single RTX 4090.

### Train on Multiple GPUs.

We use the Hugging Face Accelerate library to train on multiple GPUs. Here's an example using 4 GPUs:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --model_name=Llama
```

This speeds up training by 4×. The code also supports multi-node training, such as 32 GPUs across 4 nodes.

### Results

<img src="https://github.com/user-attachments/assets/4df41f0b-3f40-4bde-818c-3b301a9963cb" width="800">

## 2. Sample

```python
CUDA_VISIBLE_DEVICES=0 python sample.py --model_name=Llama --ckpt_path="checkpoints/train/Llama/step=10000.pth"
```

The sampled texts look like:

<pre>
We may! though a bald prove. We three, I say! What                    
must I see so, most heart?

Servant:
He hath ribbons of an the city, which he main for her
voices of the same winder. What say you to yours?

Provost:
It was commanded so willingly I do at ever.
So fortune
</pre>

## External Links

[1] Lit-Llama: https://github.com/Lightning-AI/lit-llama

[2] NanoGPT: https://github.com/karpathy/nanoGPT
