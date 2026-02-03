#!/usr/bin/env bash

TARGET=$1
MODEL=$2
CKPT_ROOT=${3:-./checkpoints}  # Default to "checkpoints" if CKPT_ROOT is not set by the user
LOG_DIR=${4:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

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

experiment_config_group=${targets_configs[${TARGET}]}
# NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.
# shellcheck disable=SC2016,SC2086
gnn-eval -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
  +experiment=persevere/"${experiment_config_group}"/"${MODEL}" \
  ${targets_overrides[${TARGET}]} \
  data/dataset/target="${TARGET}" data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
  ckpt_path="${CKPT_ROOT}/PERSEVERE/${TARGET}/${MODEL}/kfold/"'${data.split_idx}'"/best.ckpt" \
  >>"${LOG_DIR}/persevere-eval-gnn-${TARGET}-${MODEL}.log" 2>&1
