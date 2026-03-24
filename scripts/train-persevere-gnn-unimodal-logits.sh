#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

declare -A targets_configs
targets_configs=(
  # Configurations for which optimized hparams are available,
  # in which case we use the optimized hparams for each target
  [risk_ESC-2014_unimodal_logits]="risk_ESC-2014"
)
declare -A targets_overrides
targets_overrides=(
  # Override options for target configs not meant to use unimodal logits head out-of-the-box
  [risk_ESC-2014]="model/head=unimodal_logits ~model.head.num_layers"
)
models=(
#  "gcn" "gat" "gin"
#  "gcn+vn" "gat+vn" "gin+vn"
  # Different formulation of virtual nodes, with heterogeneous message passing between real and virtual nodes
#  "gcn+vn_g" "gat+vn_g" "gin+vn_g"
#  "gcn+ef" "gat+ef" "gin+ef"
#  "gcn+vn+ef" "gat+vn+ef" "gin+vn+ef"
#  "gcn+lf" "gat+lf" "gin+lf"
#  "gcn+vn+lf" "gat+vn+lf" "gin+vn+lf"
#  "gcn+gavn" "gat+gavn"
  "gin+gavn"
  # Different formulation of virtual nodes initialized with graph attributes, with heterogeneous message passing between
  # real and virtual nodes
#  "gcn+vn_gv2" "gat+vn_gv2" "gin+vn_gv2"
#  "gps"
#  "gps+ef" "gps+lf" "gps+xa"
  "gps+ftxa"
)

for target_config in "${!targets_configs[@]}"; do # Loop over targets and associated configs to use
  target="${targets_configs[${target_config}]}" # Get the underlying target to use for the optimized unimodal config
  for model in "${models[@]}"; do  # Loop over models
    # NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
    # not match. This way, the splits computed from the first target will be used for all subsequent targets.
    # shellcheck disable=SC2086
    gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
      logger=wandb test=True \
      experiment=persevere/"${target_config}"/"${model}" \
      ${targets_overrides[${target_config}]} \
      data/dataset/target="${target}" data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
      logger.wandb.project=GENESIS-ordinal-regression \
      >>"${LOG_DIR}/train-persevere-gnn-${target}-${model}.log" 2>&1
  done
done
