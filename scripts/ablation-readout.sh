#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Ablation study of impact of readout when using virtual node initialized with graph-level features
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=ablation/readout/gin+gavn,ablation/readout/gat+gavn,ablation/readout/gin+gavn-sum,ablation/readout/gat+gavn-sum,ablation/readout/gin+gavn-mean,ablation/readout/gat+gavn-mean data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/ablation_readout.log" 2>&1
