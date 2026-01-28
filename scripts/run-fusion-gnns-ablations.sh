#!/usr/bin/env bash

ABLATION=$1 # Name of the ablation folder under 'src/genesis/configs/experiment/ablation/' for which to run experiments
LOG_DIR=${2:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

for experiment_file in src/genesis/configs/experiment/ablation/"$ABLATION"/*; do
  experiment_cfg=$(basename "$experiment_file" .yaml)
  # NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
  # not match. This way, the splits computed from the first target will be used for all subsequent targets.
  gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
    logger=wandb test=True \
    experiment=ablation/"$ABLATION"/"$experiment_cfg" \
    data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
    task_name="ablation" >>"${LOG_DIR}/ablation-${ABLATION}-${experiment_cfg}.log" 2>&1
done
