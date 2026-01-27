#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

datasets=(
  "zinc" "molhiv"
)
models=(
  "gine+vn"
  "gine+vn+ef" "gine+vn+lf" "gine+gavn"
  "gps"
  "gps+ef" "gps+lf" "gps+xa" "gps+ftxa"
)

shopt -s globstar # Enable ** globbing, disabled by default in bash (see https://stackoverflow.com/a/62985520)

for dataset in "${datasets[@]}"; do # Loop over datasets
  for model in "${models[@]}"; do  # Loop over models
    # Set `paths.data_dir` to a unique folder per run to avoid conflicts when different models require different
    # pre-processing, and to avoid race conditions between different seeds that could use the same dataset but are
    # run in parallel. This speeds up total experiments runtime, at the cost of using more disk space.
    # shellcheck disable=SC2016,SC2086
    gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
      logger=wandb test=True \
      paths.data_dir='${.root_dir}'/data/duplicates4parallel_jobs/${model}/'${seed}' \
      experiment=generalization/"${dataset}"/"${model}" \
      'seed=range(10)' \
      >>"${LOG_DIR}/generalization-${dataset}-${model}.log" 2>&1
  done

  # Separate run for tabular models on the same datasets' global features
  baseline-tabular -m \
    logger=wandb test=True \
    experiment=generalization/"${dataset}"/tabular \
    model/model=tabpfn,xgboost \
    +model.model.ignore_pretraining_limits=true \
    'seed=range(10)' \
    >>"${LOG_DIR}/generalization-${dataset}-tabular.log" 2>&1

done
