# Minimal PyTorch implementation of GPT2 and Llama

This repository provides the minimal PyTorch implementation of GPT-2 and LLaMA. This repo simplifies the GPT-2 and LLaMA code for easier understanding and usage. This repo trains a natural language generation system with 1 million characters and converges in less than 5 minutes.

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/mini_llm
cd mini_llm

# Install Python environment
conda create --name llm python=3.10

# Activate environment
conda activate llm

# Install Python packages dependencies
bash env.sh
```

## 1. Train

```python
CUDA_VISIBLE_DEVICES=0 python train.py --model_name=Llama
```

We train the languge model on the Shakespeares dataset with 1 million characters. The training takes around 20 min to train for 10,000 steps on a single RTX4090.

![Training & Validation Loss](assets/loss.png)

### Train on Multiple GPUs.

We use Huggingface accelerate library to train the systems on multiple GPUs. train_accelerate.py just adds a few lines to train.py. Here is an example to run with 4 GPUs:

```python
accelerate launch --config_file ./train_accelerate_config.yaml train_accelerate.py --model_name=Llama
```

Then, the training can speed up by 4x times. The code can also train with multiple nodes such as 32 GPUs with 4 nodes.

You can modify the `./train_accelerate_config.yaml` to customize your training parameters.

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

## External links

This repo is benefited from the following repos.

NanoGPT: https://github.com/karpathy/nanoGPT

Lit-Llama: https://github.com/Lightning-AI/lit-llama

## License

MIT