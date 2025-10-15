#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Ablation study of primal/dual graph representation with similar models on node/edge features
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=ablation/graph_representation/dual-gin,ablation/graph_representation/dual-gin+pe,ablation/graph_representation/primal-gine+pe data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/ablation_graph_representation.log" 2>&1
