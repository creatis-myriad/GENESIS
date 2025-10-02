#!/usr/bin/env bash

export LOG_DIR=$1

# NOTE 1: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.

# NOTE 2: Joblib launcher (to run jobs in parallel) is not used here, because overriding the target (`data/dataset/target=...`)
# requires reloading the dataset (to update the target `y`) and overwriting the processed data files (in `data/PERSEVERE/processed`).
# If multiple jobs try to do this at the same time, they could corrupt the processed data files, leading to errors like
# 'EOFError' when unpickling the data. Or they could silently use the wrong target. To avoid this, we run the experiments sequentially.

# Multi-class classification risk prediction
gnn-train -m logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/gcn,persevere/gat,persevere/gin,persevere/gin+vn data/split=k_fold data/dataset/target=risk_ESC-2014 +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-multi_classification.log" 2>&1

# Binary classification elevated bio-markers/risk prediction
gnn-train -m logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/gcn,persevere/gat,persevere/gin,persevere/gin+vn data/split=k_fold data/dataset/target=risk_ESC-2014_elevated,troponin_elevated,nt-probnp_elevated,enzymes_elevated +data.on_conflict=ignore model/metrics=binary_classification 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-binary_classification.log" 2>&1
