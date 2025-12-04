#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# NOTE: Ignore conflicts on data splits (data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.

# Fit and score tabular baselines (model/model) across multiple targets (experiments) and feature sets (data/dataset/usecols)
# shellcheck disable=SC2016
baseline-tabular -m hydra/launcher=joblib hydra.launcher.n_jobs=10 \
  logger=wandb test=True \
  experiment=tabular_baseline/risk_ESC-2014,tabular_baseline/risk_ESC-2014_elevated,tabular_baseline/enzymes_elevated,tabular_baseline/troponin_elevated,tabular_baseline/nt-probnp_elevated \
  data/dataset/usecols=spesi,spesi+cardiac_biomarkers,spesi+graph_biomarkers,spesi+cardiac_biomarkers+graph_biomarkers \
  model/model=tabpfn,xgboost \
  data/split=k_fold data.on_conflict=ignore \
  'data.split_idx=range(10)' \
  'ckpt_backbone_save_dirpath="${paths.ckpt_dir}/${op:call,${op:methodcaller,upper},${hydra:runtime.choices.data/dataset}}/${hydra:runtime.choices.data/dataset/target}/${hydra:runtime.choices.model/model}/${hydra:runtime.choices.data/dataset/usecols}/${hydra:runtime.choices.data/split}/${data.split_idx}"' \
  'ckpt_backbone_save_filename="${op.ternary:${op:eq,${hydra:runtime.choices.model/model},tabpfn},model.tabpfn_fit,null}"' \
  >>"${LOG_DIR}/persevere-tabular.log" 2>&1
