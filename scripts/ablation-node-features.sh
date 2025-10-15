#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Ablation study of impact of node features when including graph-level features in virtual node
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=ablation/node_features/gin-vcn,ablation/node_features/gin-vcn+pe,ablation/node_features/gin-vcn+pe-only data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/ablation_node_features.log" 2>&1
