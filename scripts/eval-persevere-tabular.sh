#!/usr/bin/env bash

MODEL=$1
TARGET=${2:-"risk_ESC-2014"}  # Default to "risk_ESC-2014" if TARGET is not set by the user
USECOLS=${3:-spesi+cardiac_biomarkers+graph_biomarkers}  # Default to "spesi+cardiac_biomarkers+graph_biomarkers" if USECOLS is not set by the user
CKPT_ROOT=${4:-./checkpoints}  # Default to "checkpoints" if CKPT_ROOT is not set by the user
LOG_DIR=${5:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Determine (default) checkpoint filename based on the model type
case "$MODEL" in
  "tabpfn")
    ckpt_filename="model.tabpfn_fit"
    ;;
  "xgboost")
    ckpt_filename="model.pickle"
    ;;
  *)
    echo "Unsupported model: $MODEL"
    exit 1
    ;;
esac

# NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.
# shellcheck disable=SC2016
gnn-eval -m hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu \
  test=True \
  +experiment=tabular_baseline/"${TARGET}" \
  data/dataset/usecols="${USECOLS}" \
  model/model="${MODEL}" \
  data/dataset/target="${TARGET}" data/split=k_fold data.on_conflict=ignore 'data.split_idx=range(10)' \
  ckpt_path="${CKPT_ROOT}/PERSEVERE/${TARGET}/${MODEL}/${USECOLS}/kfold/"'${data.split_idx}'"/${ckpt_filename}" \
  >>"${LOG_DIR}/eval-persevere-tabular-${TARGET}-${MODEL}-${USECOLS}.log" 2>&1
