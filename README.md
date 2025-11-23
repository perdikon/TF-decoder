<div align="center">

# Synthetic Event Log Generation with TF Decoder

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Based%20on%20Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repository implements the experiments from the thesis
"Synthetic Event Log Generation Through the Joint Modeling of Event Attributes".
It provides a Transformer decoder (TF decoder)–based generative model for
event logs, along with training and evaluation pipelines built on PyTorch
Lightning and Hydra. The code supports multiple real-world process mining
datasets (e.g., sepsis, emergency, BPIC) and tools for preprocessing,
training, and assessing synthetic log quality.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/perdikon/TF-decoder
cd TF-decoder

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to train

Train model with default configuration

```bash
# train CVAE model on Sepsis
python src/train.py --config-name=train_cvae.yaml data=sepsis

# train CVAE model on BPIC2012
python src/train.py --config-name=train_cvae.yaml data=bpic

# train TF-decoder model on Sepsis
python src/train.py --config-name=train_tf_decoder.yaml data=sepsis

# train TF-decoder model on BPIC2012
python src/train.py --config-name=train_tf_decoder.yaml data=bpic

```

## How to run evaluation

``` bash

#evaluate CVAE on Sepsis

python src/eval.py data=sepsis model=cvae ckpt_path=<path/to/your_trained_model.ckpt>


#evaluate TF-decoder on Sepsis using beam search

python src/eval.py data=sepsis model=tf_decoder ckpt_path=<path/to/your_trained_model.ckpt> test_cfg = beam


```


## Configurations

This project uses [Hydra](https://hydra.cc/) for configuration. All config files live under `configs/`:

- `configs/train_tf_decoder.yaml` – main training config for the TF decoder model.
- `configs/train_cvae.yaml` – main training config for the CVAE model.
- `configs/eval.yaml` – main config for evaluation and synthetic log quality assessment.
- `configs/data/` – dataset configs (e.g. `sepsis.yaml`, `emergency.yaml`, `bpic.yaml`) controlling paths and preprocessing options.
- `configs/model/` – model configs (e.g. CVAE / TF decoder hyperparameters).
- `configs/trainer/` – Lightning trainer settings (CPU/GPU, mixed precision, logging, etc.).
- `configs/experiment/` – optional experiment configs overriding defaults for specific runs.

You can override any setting from the command line, e.g.:

```bash
python src/train.py data=sepsis trainer.max_epochs=20 model.lr=1e-4
```

