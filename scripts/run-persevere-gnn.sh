#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

declare -A targets_configs
targets_configs=(
  # Targets for which optimized hparams are available
  [risk_ESC-2014]="risk_ESC-2014"
  [enzymes_elevated]="enzymes_elevated"
  # Targets for which optimized hparams are not available,
  # in which case we default to optimized hparams from a compatible target
  [troponin_elevated]="enzymes_elevated"
  [nt-probnp_elevated]="enzymes_elevated"
)
models=(
  "mlp" "mlp+gaef" "mlp+galf"
  "gcn" "gcn+vn" "gcn+vn_g" "gcn+gaef" "gcn+galf" "gcn+gavn" "gcn+vn_gv2"
  "gat" "gat+vn" "gat+vn_g" "gat+gaef" "gat+galf" "gat+gavn" "gat+vn_gv2"
  "gin" "gin+vn" "gin+vn_g" "gin+gaef" "gin+galf" "gin+gavn" "gin+vn_gv2"
  "gps" "gps+galf" "gagps"
)

for target in "${!targets_configs[@]}"; do # Loop over targets and associated
  experiment_config_group=${targets_configs[${target}]}
  for group in "${!models[@]}"; do  # Loop over model groups
    for model in ${models[${group}]}; do  # Loop over specific models
      # NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
      # not match. This way, the splits computed from the first target will be used for all subsequent targets.
      gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
        logger=wandb logger.wandb.log_model=True test=True \
        experiment=persevere/"${experiment_config_group}"/"${model}" \
        data/dataset/target="${target}" data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' \
        >>"${LOG_DIR}/persevere-gnn-${target}-${model}.log" 2>&1
    done
  done
done
