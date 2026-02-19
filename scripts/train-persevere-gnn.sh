#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

targets=(
  "risk_ESC-2014"
)
models=(
  "mlp" "gcn" "gat" "gin"
  "gcn+vn" "gat+vn" "gin+vn"
  # Different formulation of virtual nodes, with heterogeneous message passing between real and virtual nodes
#  "gcn+vn_g" "gat+vn_g" "gin+vn_g"
  "mlp+ef" "gcn+ef" "gat+ef" "gin+ef"
  "gcn+vn+ef" "gat+vn+ef" "gin+vn+ef"
  "mlp+lf" "gcn+lf" "gat+lf" "gin+lf"
  "gcn+vn+lf" "gat+vn+lf" "gin+vn+lf"
  "gcn+gavn" "gat+gavn" "gin+gavn"
  # Different formulation of virtual nodes initialized with graph attributes, with heterogeneous message passing between
  # real and virtual nodes
#  "gcn+vn_gv2" "gat+vn_gv2" "gin+vn_gv2"
  "gps"
  "gps+ef" "gps+lf" "gps+xa" "gps+ftxa"
)

for target in "${targets[@]}"; do # Loop over targets and associated hparams to use
  for model in "${models[@]}"; do  # Loop over models
    # NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
    # not match. This way, the splits computed from the first target will be used for all subsequent targets.
    gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
      logger=wandb test=True \
      experiment=persevere/"${target}"/"${model}" \
      data/dataset/target="${target}" data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
      >>"${LOG_DIR}/train-persevere-gnn-${target}-${model}.log" 2>&1
  done
done
