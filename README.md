<div align="center">

# GENESIS

Code repository for the Graph nEural Networks for pulmonary EmboliSm rIsk Stratification (GENESIS) project.

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/pytorch-lightning)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![lightning-hydra-template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/nathanpainchaud/lightning-hydra-template)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<br>
[![code-quality](https://github.com/creatis-myriad/GENESIS/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/creatis-myriad/GENESIS/actions/workflows/code-quality-main.yaml)
[![tests](https://github.com/creatis-myriad/GENESIS/actions/workflows/tests.yaml/badge.svg)](https://github.com/creatis-myriad/GENESIS/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/creatis-myriad/GENESIS/branch/main/graph/badge.svg)](https://codecov.io/gh/creatis-myriad/GENESIS)
<br>
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/creatis-myriad/GENESIS?tab=Apache-2.0-1-ov-file)

# Publications

</div>

## Description

A project applying tabular models and graph neural networks to the task of pulmonary embolism risk stratification.

For tabular models, it is assumed that a CSV of global features (i.e., medical records, cardiac biomarkers, vascular biomarkers)
for all patients is available. This code repository can then fit tabular models to predict the risk of pulmonary embolism
from these features.

For graph neural networks (GNNs), it is assumed that an image processing pipeline previously segmented and extracted the
graph of the vascular tree from 3D CTPA images. This code repository then takes these graphs as input and trains GNNs
to predict the risk of pulmonary embolism from the vascular graphs and global features.

> [!IMPORTANT]
> Using this project requires a basic understanding of PyTorch Lightning and Hydra. If you do not know at least what
> these libraries do and how they work at a high level, you should familiarize yourself with them.
> We refer you to the [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) and the
> [Hydra documentation](https://hydra.cc/docs/intro/).

### Table of Contents

1. [Installation](#installation)
   1. [`uv`](#uv-recommended)
   2. [`pip`](#pip)
   3. [Extras](#list-of-available-extras)
   4. [Weight & Biases configuration](#setup-weight--biases)
2. [Reproduce published experiments](#reproduce-published-experiments)
3. [Run custom experiments](#run-custom-experiments)
   1. [Basics](#the-basics)
   2. [Preset configs](#use-preset-configs)
   3. [Track experiments](#track-experiments)
   4. [Launch multiple experiments simultaneously](#run-multiple-experiments)
   5. [Hyperparameter search with Optuna](#run-automatic-hyperparameter-search-with-optuna)
4. [Run tests](#run-tests)

## Installation

#### uv (recommended)

> [!NOTE]
> [uv](https://docs.astral.sh/uv/) is a Python package and project manager.
> It allows you to manage Python interpreters, dependencies, and project configuration in a single tool.
> If you don't have it installed already, you can install it (on Linux and macOS) by running:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

1. Download the repository.
   ```bash
   git clone https://github.com/creatis-myriad/GENESIS
   cd GENESIS
   ```
2. Create a virtual environment and install the project and its dependencies. You must specify as an extra the desired
   compute platform for PyTorch (i.e. CPU/CUDA). Supported values are: `cpu`, `cu129`, `cu128`, `cu126`.
   ```bash
   # e.g. to install the project with the PyTorch version built for CPU
   uv sync --extra cpu

   # e.g. to install the project with the PyTorch version built for CUDA 12.8
   uv sync --extra cu128
   ```
   [OPTIONAL] You can also specify other extras for additional functionalities:
   ```bash
   # e.g. to install the `wandb` extra for W&B integration
   uv sync --extra cpu --extra wandb

   # e.g. to install all extra functionalities at once
   uv sync --extra cpu --extra all
   ```
3. Activate the virtual environment created by `uv`.
   ```bash
   source .venv/bin/activate
   ```

#### Pip

1. Download the repository.
   ```bash
   git clone https://github.com/creatis-myriad/GENESIS
   cd GENESIS
   ```
2. Create a virtual environment and activate it.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/).
   Follow the instructions for `pip` and the compute platform compatible with your system.
   ```bash
   # e.g. to install the PyTorch version built for CPU
   pip install torch --index-url https://download.pytorch.org/whl/cpu

   # e.g. to install the PyTorch version built for CUDA 12.8
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
4. Install PyG and its `torch_scatter` dependency according to the [official instructions](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html)
   Follow the instructions for `pip` and the compute platform compatible with your system.
   ```bash
   # install PyG
   pip install torch_geometric

   # install `torch_scatter` optional dependency (e.g. for CUDA 12.8)
   pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
   ```
5. Install the project in editable mode.
   ```bash
   pip install -e .
   ```
   [OPTIONAL] You can also specify other extras for additional functionalities:
   ```bash
   # e.g. to install the `wandb` extra for W&B integration
   pip install -e .[wandb]

   # e.g. to install all extra functionalities at once
   pip install -e .[all]
   ```

### List of available extras

- \[`cpu`|`cu129`|`cu128`|`cu126`\]: Required mutually exclusive extras to install the project with a PyTorch version
  built for CPU or a specific CUDA version (only available when using `uv`, not `pip`).
- `all`: Install all (non-mutually exclusive) extras at once.
- `baselines`: Extra dependencies required to run the baselines.
- `totalsegmentator`: For using the pretrained `TotalSegmentator` model for segmenting heart ventricles to preprocess images.
- `wandb`: For experiment tracking with Weights & Biases.

### Setup Weight & Biases

#### Create an account

Follow the instructions on the [Weights & Biases website](https://docs.wandb.ai/quickstart#1-create-an-account-and-install-wb)
to create an account.

#### Install W&B

Make sure that you install the `wandb` extra when installing the project, as shown in the [installation instructions](#installation).

#### Configure your credentials

The recommended way to configure your W&B credentials is to expose them as environment variables
(see [W&B's documentation on this](https://docs.wandb.ai/guides/track/environment-variables/)). You can do this by copying the
[`configs/local/example.yaml`](src/genesis/configs/local/example.yaml) to a new `default.yaml` (which will be ignored by
Git) and filling in your W&B credentials.

You don't have to do anything more than that, as the project is configured to automatically load keys under
`hydra.job.env_set` as environment variables when executing the scripts.

#### Use the wandb logger

Follow the instructions provided in the [How to run](#track-experiments) section to enable experiment tracking via W&B.

## Reproduce published experiments

The commands below are meant to reproduce the experiments described in the paper. They will run different combinations
of models and data configurations in a 10-fold cross-validation setting.

Results are logged both locally and online on W&B (see [previous section for instructions to set up W&B](#setup-weight--biases)).
Each experiment corresponds to a configuration run on a specific cross-validation fold. To facilitate analysis, groups
in W&B correspond to the same configuration run on the cross-validation folds.

Depending on the type of model (tabular or GNN) different Python entry point scripts are called:

- Tabular models use the [`tabular_baseline.py`](src/genesis/tabular_baseline.py) script;
- GNNs use the [`train.py`](src/genesis/train.py) script.

> [!WARNING]
> Running the experiments below requires access to the PERSEVERE dataset, which is not publicly available.
> Thus, the scripts should not be expected to run as-is without the dataset. Rather, the scripts and code are provided
> for reference.

> [!TIP]
> Since models are implemented in a dataset-agnostic way, implementing PyG datasets and providing corresponding configs
> should be all that is needed to test the models on other datasets.

### Ablation study of global features with tabular models for risk stratification

To run tabular models (TabPFN, XGBoost) on combinations of global features (medical records, cardiac biomarkers, vascular biomarkers):

```bash
scripts/train-persevere-tabular.sh
```

### Benchmark of GNNs on vascular graph and global features for risk stratification

To run GNN backbones (GCN, GAT, GIN, GPS), with and without Virtual Nodes (VN) for MPNN backbones, and with different
strategies to combine global features (early fusion (EF), late fusion (LF), virtual node (VN), Feature Tokenizer with cross-attention (FTxA)):

```bash
scripts/train-persevere-gnn.sh
```

### Vascular biomarkers regression as sanity check on GNNs

To compare the best tabular and GNN backbones for the prediction of vascular biomarkers that are derived from local graph features:

- Runs TabPFN on global features (medical records, cardiac biomarkers);
- Runs GIN and GPS on the vascular graphs.

```bash
scripts/run-persevere-sanity-check-targets.sh
```

### Ablation study of data and graph representations on the best GNN configuration

To run the best GNN configuration with alternative graph and global features representations:

```bash
# Test the primal graph representation.
# The default config uses the dual (i.e. line graph) representation.
scripts/run-persevere-gnn-ablation.sh graph_representation

# Test linear and TabPFN embedding of global features.
# The default config uses the Feature Tokenizer embedding.
scripts/run-persevere-gnn-ablation.sh global_features_embedding

# Test using a CLS token on global features as readout, i.e. graph-level representation.
# The default configuration uses global graph pooling (i.e., mean or sum depending on the config).
scripts/run-persevere-gnn-ablation.sh readout
```

> [!IMPORTANT]
> The results of these runs are meant to be compared to runs launched with the best GNN configuration,
> GPS + Feature Tokenizer with Cross-Attention (gps+ftxa), run as part of the [GNN benchmark](#benchmark-of-gnns-on-vascular-graph-and-global-features-for-risk-stratification).

> [!TIP]
> Calling the [`run-persevere-gnn-ablation.sh`](scripts/run-persevere-gnn-ablation.sh) script with the name of one of
> the folders in [`configs/experiment/ablation`](src/genesis/configs/experiment/ablation) will run all the experiment
> configs in that folder using the [`train.py`](src/genesis/train.py) script.

## Run custom experiments

This section describes how to configure individual experiments, e.g., to change hyperparameters, models, datasets, etc.,
if you want more control over the configuration than the predefined batch of experiments described in the [previous section](#reproduce-published-experiments).

### The basics

Train model with the default configuration (on the small MUTAG dataset).

```bash
# train on CPU
gnn-train trainer=cpu

# train on GPU
gnn-train trainer=gpu
```

Override any individual parameter in the config files from the command line like this:

```bash
# override the number of epochs and batch size
gnn-train trainer.max_epochs=20 data.batch_size=64 ...

# train default model on your dataset
gnn-train data/dataset=<YOUR_DATASET_CONFIG> ...

# train your model on the default dataset
gnn-train model=<YOUR_MODEL_CONFIG> ...
```

To evaluate a trained model, use the [`gnn-eval`](src/genesis/eval.py) script.

```bash
# evaluate a trained model on your dataset's test set
gnn-eval data=<DATAMODULE_CONFIG> data/dataset=<DATASET_CONFIG> model=<MODEL_CONFIG> ckpt_path=<PATH_TO_CHECKPOINT>
```

### Use preset configs

Train model with chosen experiment configuration from [configs/experiment/](src/genesis/configs/experiment/).

> [!TIP]
> This allows you to provide (complete) presets on top of the default configuration, typically for experiments you want
> to run regularly.

```bash
gnn-train experiment=<YOUR_EXPERIMENT_CONFIG>
```

### Track experiments

The implemented tool to track experiments is [Weights & Biases](https://wandb.ai/site), by using W&B's
[integration in PyTorch Lightning](https://docs.wandb.ai/guides/integrations/lightning/).

> [!WARNING]
> You must have followed the [W&B setup instructions](#setup-weight--biases) to use this feature.

```bash
# track experiment online w/ W&B
gnn-train logger=wandb

# track experiment offline w/ W&B
gnn-train logger=wandb logger.wandb.offline=True
```

### Run multiple experiments

Launch multiple experiments at once using the `multirun` (`-m`) option.

```bash
# run multiple experiments sequentially, here w/ 5 different seeds
gnn-train -m seed=0,1,2,3,4
```

Launch multiple experiments at once **in parallel** using the [Joblib launcher for Hydra](https://hydra.cc/docs/plugins/joblib_launcher/).

> [!NOTE]
> The `hydra-joblib-launcher` plugin required to use this feature is installed by default with the project, so no need
> to install it by yourself.

```bash
# run multiple experiments in parallel, here w/ 5 different seeds
gnn-train -m hydra/launcher=joblib seed=0,1,2,3,4
```

### Run automatic hyperparameter search with Optuna

Launch an automatic hyperparameter search using the [Optuna sweeper for Hydra](https://hydra.cc/docs/plugins/optuna_sweeper/).

> [!WARNING]
> You have to make sure that the `hparams_search` config you use is compatible with the model, since `hparams_search`
> defines how to sweep over model-dependent config options.

```bash
# Example of a predefined Optuna config for graph-level models compatible with the default experiment
gnn-train hparams_search=graph_classification_optuna
```

> [!TIP]
> Optuna can be used in a cross-validation setting, by evaluating each sampling of hyperparameters on the different
> dataset folds and reporting the average performance. However, this approach is not compatible with the default Optuna
> sweeper plugin, where each trial corresponds to one Hydra run, i.e. one model trained/evaluated on a specific
> partition of the dataset.
>
> To support this feature, we rely on our custom `serial_sweeper`, designed to run multiple jobs in sequence within the
> same Hydra run and then aggregate the results of these jobs. By sweeping over the different folds with this sweeper,
> we support cross-validation with Optuna.
>
> This is all handled already in the predefined Optuna config `graph_classification_optuna` for graph-level models.
> If you want to support this in your own Optuna config, all you have to do is to use the predefined `splits` config for
> `serial_sweeper`, and make sure that `data/split=kfold` is used to split the data into multiple folds.
>
> ```bash
> gnn-train [...] hparams_search=<YOUR_OPTUNA_CONFIG> data/split=k_fold serial_sweeper=splits
> ```

## Run tests

Run the tests using [Pytest](https://docs.pytest.org/en/stable/).

```bash
# run all tests
pytest

# run a test package
pytest tests/integration

# run tests from a specific file
pytest tests/integration/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```
