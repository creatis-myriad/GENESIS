#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Map datasets to their root folder names under the data root directory
declare -A datasets
datasets=(
  [zinc]="ZINC"
  [molhiv]="ogbg_molhiv"
)
models=(
  "gine+vn" "gine+vn+ef" "gine+vn+lf" "gine+gavn"
  "gps" "gps+ef" "gps+lf" "gps+xa" "gps+ftxa"
)

shopt -s globstar # Enable ** globbing, disabled by default in bash (see https://stackoverflow.com/a/62985520)

for dataset in "${!datasets[@]}"; do # Loop over datasets
  for model in "${models[@]}"; do  # Loop over models
    # Clear cached processed data manually, instead of using `data.dataset.force_reload=True` option
    # This has the same effect as `force_reload` of avoiding errors between models requiring different pre-processing,
    # while avoiding re-processing datasets between different random seeds for the same model.
    rm -rf ./data/"${datasets[${dataset}]}"/**/processed
    gnn-train -m trainer=gpu data.num_workers=6 \
      logger=wandb test=True \
      experiment=generalization/"${dataset}"/"${model}" \
      'seed=range(10)' \
      >>"${LOG_DIR}/generalization-${dataset}-${model}.log" 2>&1
  done
done
