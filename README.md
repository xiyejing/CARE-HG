# CARE-HG

**Debiased Causal Reasoning Hypergraphs for Multimodal Emotion-Cause Pair Extraction in Conversations**

This repository contains the source code for CARE-HG. The framework uses context and prior hypergraphs to model utterance-level relations, then combines factual and counterfactual branches for emotion, cause, and emotion-cause pair prediction. At inference time, the current implementation debiases emotion scores by subtracting counterfactual logits from factual logits.

## Method overview

~~~mermaid
flowchart LR
    A[Dialogue text, audio, video] --> B[Utterance representations]
    B --> C[Context hypergraph]
    D[LLM-derived prior knowledge] --> E[Prior hypergraph]
    C --> F[Factual branch]
    E --> F
    C --> G[Counterfactual branch]
    E --> G
    F --> H[Emotion, cause, and EC-pair predictions]
    G --> I[Counterfactual emotion scores]
    H --> J[Debiased emotion scores]
    I --> J
    J --> K[Emotion prediction]
~~~

The overview follows the supplied framework figure. The repository snapshot does not currently include that image as a tracked asset.

## Repository layout

~~~text
CARE-HG/
├── run.py
├── models/
│   ├── CME.py
│   ├── DialogueRNN.py
│   ├── GAT.py
│   ├── MultiHeadAttention.py
│   ├── PairNN.py
│   ├── Transformer.py
│   ├── UME.py
│   ├── model.py
│   └── model_hyper.py
└── utils/
    ├── data_loader.py
    └── eval_func.py
~~~

## Environment

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

## Data and checkpoints

The default configuration values in **run.py** expect data and model files outside the repository:

~~~text
data/
├── ECF/
│   ├── prior_knowledge.json
│   └── videos/
└── MECAD/
    └── prior_knowledge1.json

pretrained_models/
├── textual/RoBERTa/
├── audio/wav2vec2-base-960h/
└── video/XCLIP/
~~~

- The default dataset is ECF. The training data root is formed from **data_dir** and **data_name**.
- **models/model_hyper.py** currently uses hard-coded prior paths: **data/ECF/prior_knowledge.json** for ECF and **data/MECAD/prior_knowledge1.json** for MECAD.
- The RoBERTa checkpoint directory must contain the Transformers configuration/tokenizer files and the checkpoint named by **init_pretrain_checkpoint** (default: **pytorch_model.bin**).
- Audio and video feature pickle paths can be set with **audio_features_pkl_path** and **video_features_pkl_path**. Their exact schemas are defined in the dataset preprocessing code, which is not present in the current repository snapshot.
- The default modality list is text plus audio. Video can be enabled through the configuration when its features and compatible model weights are available.

## Configuration and running

The entry point accepts a YAML file through **--config_file** and defaults to **config.yaml**. That file is not currently checked in. A minimal example based on the defaults in **run.py** is:

~~~yaml
data_dir: data
data_name: ECF
network: MECPEC

textual_pretrain_model_dir: pretrained_models/textual/RoBERTa
audio_pretrain_model_dir: pretrained_models/audio/wav2vec2-base-960h
video_pretrain_model_dir: pretrained_models/video/XCLIP
video_dir: data/ECF/videos
init_pretrain_checkpoint: pytorch_model.bin
output_dir: output

pretrain_encoder_type: RoBERTa
modality: [textual, audio]
use_llm: true

do_train: true
do_eval: true
seed: [67137]
num_train_epochs: 50
train_batch_size: 64
eval_batch_size: 64
learning_rate: 0.00005
gradient_accumulation_steps: 2
~~~

Save the configuration as **config.yaml**, then run:

~~~bash
python run.py --config_file config.yaml
~~~

With both **do_train** and **do_eval** enabled, the intended flow is training, validation, selection of **model_best.pt**, and test evaluation. The model name used by the main path is **MECPEC**; **RankCP** is also referenced in **run.py**.

The current entry point iterates over **config.seed** but calls **run(config, 67137)** each time. As written, changing the seed list does not change the seed passed to training.

## LLM prior knowledge and prompt template

The current code does **not** call an LLM service or store a verbatim prompt. Instead, the model loads a precomputed prior-knowledge JSON file. For each conversation, it looks up each 1-based utterance ID and uses the maximum confidence value in that utterance's prior entries as a bias for the causal mask.

The following is a suggested prompt for generating candidate priors. It is a template based on the framework figure, not a prompt recovered from the repository:

~~~text
You are annotating emotion-cause pairs in a dialogue.

Use the supplied emotion label set and the 1-based utterance IDs.
For each emotion-bearing utterance, identify the utterance or utterances
that provide evidence for its emotion. A cause may occur before or after
the emotion utterance. Include only pairs supported by the dialogue.

Emotion labels:
{emotion_label_set}

Dialogue:
{dialogue_with_1_based_utterance_ids}

Return only a JSON array. Each item must contain:
- EmoID: the 1-based ID of the emotion utterance
- CauID: the 1-based ID of a cause utterance
- Conf: confidence for this pair, as a number from 0 to 1

Example:
[
  {"EmoID": 1, "CauID": 3, "Conf": 0.92}
]

Do not include explanations, markdown, or fields beyond EmoID, CauID, and Conf.
~~~

The model-side prior lookup is keyed by conversation ID, then by the string form of the 1-based emotion utterance ID. A compatible nested representation is:

~~~json
{
  "dialogue_001": {
    "1": {
      "3": 0.92
    }
  }
}
~~~

Confirm conversation IDs and any dataset-specific conversion from the prompt output to this nested structure against the dataset code before generating priors at scale.

## Outputs

The training script writes model checkpoints beneath **output/<data_name>_<pretrain_encoder_type>/**, including per-epoch checkpoints and **model_best.pt**. Test predictions and metrics are written to a seed-named text file. Preprocessed dataset caches are written beneath **processed/**, and the script also initializes offline Weights & Biases logging and TensorBoard logs.

The evaluation code reports emotion, cause, and emotion-cause pair metrics, including weighted F1 summaries.

## Current release limitations

A clean checkout of the current **main** branch is not runnable yet. The files below are referenced by imports but are absent from the repository tree:

- **config.yaml**
- **utils/dataset.py**
- **utils/global_variables.py**
- **model/model_hyper_CF_4edge_multimodal.py**
- **model/model_mecad.py**
- **model/baselines/rankcp/rank_cp.py**

The checked-in model files are under **models/**, while their imports use the singular **model** package. Restore the missing modules or align the imports and package layout before running the command above.

The video path also needs verification: **run.py** constructs a **VideoMAEModel**, while the default checkpoint directory is named **XCLIP**. Use a checkpoint compatible with the selected Transformers model class.

## Reference

The README structure follows the [DDIB repository](https://github.com/xiyejing/DDIB) as a style reference.
