#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

declare -A targets_configs
targets_configs=(
  # Targets for which optimized hparams are not available,
  # in which case we default to optimized hparams from the main target
  [qanadli]="risk_ESC-2014"
  [mastora_central]="risk_ESC-2014"
  [mastora_peripheral]="risk_ESC-2014"
  [total_embolism_volume]="risk_ESC-2014"
)
declare -A targets_overrides
targets_overrides=(
  # Override metrics to regression for continuous targets
  [qanadli]="model/metrics=regression"
  [mastora_central]="model/metrics=regression"
  [mastora_peripheral]="model/metrics=regression"
  [total_embolism_volume]="model/metrics=regression"
)
models=(
  # Test unimodal GNNs only, without global features, to evaluate the predictive ability of the graph alone
  "mlp"
  "gcn+vn" "gat+vn" "gin+vn"
  "gps"
)

for target in "${!targets_configs[@]}"; do # Loop over targets and associated hparams to use
  experiment_config_group=${targets_configs[${target}]}
  for model in "${models[@]}"; do  # Loop over models
    # NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
    # not match. This way, the splits computed from the first target will be used for all subsequent targets.
    # shellcheck disable=SC2086
    gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
      logger=wandb test=True \
      experiment=persevere/"${experiment_config_group}"/"${model}" \
      ${targets_overrides[${target}]} \
      data/dataset/target="${target}" data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
      >>"${LOG_DIR}/run-persevere-sanity-check-targets-${target}-${model}.log" 2>&1
  done

  # Separate run for tabular models on the same target
  # shellcheck disable=SC2086
  baseline-tabular -m \
    logger=wandb test=True \
    experiment=tabular_baseline \
    model/model=tabicl,tabpfn,xgboost \
    data/dataset/usecols=spesi+cardiac_biomarkers \
    ${targets_overrides[${target}]} \
    data/dataset/target="${target}" data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
    >>"${LOG_DIR}/run-persevere-sanity-check-targets-${target}-tabular.log" 2>&1
done
