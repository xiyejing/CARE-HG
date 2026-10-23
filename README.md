# CARE-HG

**Debiased Causal Reasoning Hypergraphs for Multimodal Emotion-Cause Pair Extraction in Conversations**

This repository contains the source code for CARE-HG. The framework uses context and prior hypergraphs to model utterance-level relations, then combines factual and counterfactual branches for emotion, cause, and emotion-cause pair prediction. At inference time, the current implementation debiases emotion scores by subtracting counterfactual logits from factual logits.

## 🧠 Method Overview

<p align="center">
  <img src="assets/framework.png" alt="CARE-HG framework" width="100%">
</p>

## 🛠️ Environment

The repository does not include a requirements file or a pinned Python, PyTorch, Transformers, or CUDA environment. The following is a starting point; choose PyTorch and DGL builds that match each other and your CUDA runtime.

~~~bash
conda create -n care-hg python=3.10 -y
conda activate care-hg

# Install the PyTorch build for your operating system and compute platform:
# https://pytorch.org/get-started/locally/

# Install DGL for the selected PyTorch/CUDA combination:
# https://www.dgl.ai/pages/start.html

pip install numpy scipy scikit-learn pyyaml tqdm wandb tensorboardX transformers
~~~

The checked-in Python files import PyTorch, DGL, Transformers, NumPy, SciPy, scikit-learn, PyYAML, tqdm, Weights & Biases, and tensorboardX. Exact package versions should be recorded when a verified environment is available.

## 🚀 Training

Set the dataset and pretrained model paths in `config.yaml`. The example below enables training and skips test-set evaluation. The training path still constructs a validation loader, so `eval_batch_size` is used during training.

~~~yaml
data_dir: data
data_name: ECF
network: MECPEC

textual_pretrain_model_dir: pretrained_models/textual/RoBERTa
audio_pretrain_model_dir: pretrained_models/audio/wav2vec2-base-960h
init_pretrain_checkpoint: pytorch_model.bin
output_dir: output

pretrain_encoder_type: RoBERTa
modality: [textual, audio]
use_llm: true

do_train: true
do_eval: false
seed: [67137]
num_train_epochs: 50
train_batch_size: 64
eval_batch_size: 64
learning_rate: 0.00005
gradient_accumulation_steps: 2
~~~

Save this configuration as `config.yaml`, then start training:

~~~bash
python run.py --config_file config.yaml
~~~
