#!/usr/bin/env bash

DEFAULT_HPARAMS=${1:-"enzymes_elevated"}  # Default to "enzymes_elevated" for targets for which optimized hparams are not available
LOG_DIR=${2:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.

# Targets for which optimized hparams are available
for target in risk_ESC-2014 enzymes_elevated; do
  gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=5 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/${target}/gcn,persevere/${target}/gcn+vn,persevere/${target}/gcn-vcn,persevere/${target}/gat,persevere/${target}/gat+vn,persevere/${target}/gat-vcn,persevere/${target}/gin,persevere/${target}/gin+vn,persevere/${target}/gin-vcn,persevere/${target}/gps data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-${target}.log" 2>&1
done

# For other (binary classification) targets, default to configured fallback hparams
for target in troponin_elevated nt-probnp_elevated; do
  gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=5 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/"${DEFAULT_HPARAMS}"/gcn,persevere/"${DEFAULT_HPARAMS}"/gcn+vn,persevere/"${DEFAULT_HPARAMS}"/gcn-vcn,persevere/"${DEFAULT_HPARAMS}"/gat,persevere/"${DEFAULT_HPARAMS}"/gat+vn,persevere/"${DEFAULT_HPARAMS}"/gat-vcn,persevere/"${DEFAULT_HPARAMS}"/gin,persevere/"${DEFAULT_HPARAMS}"/gin+vn,persevere/"${DEFAULT_HPARAMS}"/gin-vcn,persevere/"${DEFAULT_HPARAMS}"/gps data/split=k_fold data/dataset/target=${target} model/metrics=binary_classification +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-${target}.log" 2>&1
done
