# CARE-HG

**Debiased Causal Reasoning Hypergraphs for Multimodal Emotion-Cause Pair Extraction in Conversations**

Yejing Xi, Wenlong Dong, Qinglin Chang, and Qirong Mao

> Multimodal Emotion-Cause Pair Extraction (MECPE) in conversations aims to identify emotions and their corresponding causes from conversational data across text, audio, and video modalities. A key yet underexplored challenge in MECPE is that emotions are often associated with multiple correlated cues, while only a subset of them are causally necessary. Existing methods heavily rely on correlation-based learning and thus struggle to effectively distinguish statistical associations from genuine emotion causal effects, leading to spurious associations. To address this, we propose debiased CAusal REasoning HyperGraph named CARE-HG, a causality-aware framework for emotion-cause analysis. Specifically, we construct an LLM-guided heterogeneous hypergraph that leverages LLM priors as structural guidance to model interactions between multimodal emotion cues and potential causes. Furthermore, we introduce a novel prior-aware counterfactual intervention mechanism that masks salient modality nodes as potential causal factors, enabling robust learning of causal invariances. Extensive experiments on benchmark MECPE datasets demonstrate that CARE-HG consistently outperforms state-of-the-art methods, validating its effectiveness for debiased multimodal emotion-cause reasoning. Our code is available at [the CARE-HG GitHub repository](https://github.com/xiyejing/CARE-HG).

<p align="center">
  <img src="assets/framework.png" alt="CARE-HG framework" width="100%">
</p>

## 🛠️ Environment


~~~bash
conda create -n care-hg python=3.10 -y
conda activate care-hg

# PyTorch installation guide:
# https://pytorch.org/get-started/locally/

# DGL installation guide:
# https://www.dgl.ai/pages/start.html

pip install numpy scipy scikit-learn pyyaml tqdm wandb tensorboardX transformers
~~~


## 🚀 Training

Set the dataset and pretrained model paths in `config.yaml`.

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


