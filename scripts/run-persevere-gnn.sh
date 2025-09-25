#!/usr/bin/env bash

export LOG_DIR=$1

# NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.

# Multi-class classification risk prediction
gnn-train -m logger=wandb trainer=gpu test=True experiment=persevere/default data/split=k_fold data/dataset/target=risk_ESC-2014 +data.on_conflict=ignore model/components@model.encoder=gin,gat,gcn 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-multi_classification.log" 2>&1

# Binary classification elevated bio-markers/risk prediction
gnn-train -m logger=wandb trainer=gpu test=True experiment=persevere/default data/split=k_fold data/dataset/target=risk_ESC-2014_elevated,troponin_elevated,nt-probnp_elevated,enzymes_elevated +data.on_conflict=ignore model/metrics=binary_classification model/components@model.encoder=gin,gat,gcn 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-binary_classification.log" 2>&1
