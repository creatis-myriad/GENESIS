#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

declare -A targets_configs
targets_configs=(
  # Targets for which optimized hparams are available
  [risk_ESC-2014]="risk_ESC-2014"
  [enzymes_elevated]="enzymes_elevated"
  # Targets for which optimized hparams are not available,
  # in which case we default to optimized hparams from a compatible target
  [risk_ESC-2014_elevated]="risk_ESC-2014"
  [troponin_elevated]="enzymes_elevated"
  [nt-probnp_elevated]="enzymes_elevated"
)
declare -A targets_overrides
targets_overrides=(
  # Since the binary `risk_ESC-2014_elevated` targets uses the config from the multiclass `risk_ESC-2014` target,
  # it must override multiclass metrics config with binary config
  [risk_ESC-2014_elevated]="model/metrics=binary_classification"
)
models=(
  "mlp" "gcn" "gat" "gin" "gps"
  "gcn+vn" "gat+vn" "gin+vn"
  "gcn+vn_g" "gat+vn_g" "gin+vn_g"
  "mlp+gaef" "gcn+gaef" "gat+gaef" "gin+gaef"
  "mlp+galf" "gcn+galf" "gat+galf" "gin+galf" "gps+galf"
  "gcn+gavn" "gat+gavn" "gin+gavn"
  "gcn+vn_gv2" "gat+vn_gv2" "gin+vn_gv2"
  "gagps"
)

for target in "${!targets_configs[@]}"; do # Loop over targets and associated
  experiment_config_group=${targets_configs[${target}]}
  for model in "${models[@]}"; do  # Loop over models
    # NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
    # not match. This way, the splits computed from the first target will be used for all subsequent targets.
    # shellcheck disable=SC2086
    gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
      logger=wandb logger.wandb.log_model=True test=True \
      experiment=persevere/"${experiment_config_group}"/"${model}" \
      ${targets_overrides[${target}]} \
      data/dataset/target="${target}" data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' \
      >>"${LOG_DIR}/persevere-gnn-${target}-${model}.log" 2>&1
  done
done
