#!/usr/bin/env bash

LOG_DIR=${LOG_DIR:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# NOTE 1: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.

# Multi-class classification risk prediction
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=5 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/gcn,persevere/gat,persevere/gin,persevere/gin+vn,persevere/gin-vcn,persevere/gps data/split=k_fold data/dataset/target=risk_ESC-2014 +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-multi_classification.log" 2>&1

# Binary classification elevated bio-markers/risk prediction
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=5 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/gcn,persevere/gat,persevere/gin,persevere/gin+vn,persevere/gin-vcn,persevere/gps data/split=k_fold data/dataset/target=risk_ESC-2014_elevated,troponin_elevated,nt-probnp_elevated,enzymes_elevated +data.on_conflict=ignore model/metrics=binary_classification 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-binary_classification.log" 2>&1
