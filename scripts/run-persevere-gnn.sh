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
)
declare -A targets_overrides
targets_overrides=(
  # Since the binary `risk_ESC-2014_elevated` targets uses the config from the multiclass `risk_ESC-2014` target,
  # it must override multiclass metrics config with binary config
  [risk_ESC-2014_elevated]="model/metrics=binary_classification"
)
models=(
  "mlp" "gcn" "gat" "gin"
  "gcn+vn" "gat+vn" "gin+vn"
  # Different formulation of virtual nodes, with heterogeneous message passing between real and virtual nodes
#  "gcn+vn_g" "gat+vn_g" "gin+vn_g"
  "mlp+ef" "gcn+ef" "gat+ef" "gin+ef"
  "mlp+lf" "gcn+lf" "gat+lf" "gin+lf" "gps+lf"
  "gcn+gavn" "gat+gavn" "gin+gavn"
  # Different formulation of virtual nodes initialized with graph attributes, with heterogeneous message passing between
  # real and virtual nodes
#  "gcn+vn_gv2" "gat+vn_gv2" "gin+vn_gv2"
  "gps"
  "gps+ef" "gagps" "gps+ftga"
)

for target in "${!targets_configs[@]}"; do # Loop over targets and associated hparams to use
  experiment_config_group=${targets_configs[${target}]}
  for model in "${models[@]}"; do  # Loop over models
    # NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
    # not match. This way, the splits computed from the first target will be used for all subsequent targets.
    # shellcheck disable=SC2086
    gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
      logger=wandb test=True \
      experiment=persevere/"${experiment_config_group}"/"${model}" \
      ${targets_overrides[${target}]} \
      data/dataset/target="${target}" data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' \
      >>"${LOG_DIR}/persevere-gnn-${target}-${model}.log" 2>&1
  done
done
