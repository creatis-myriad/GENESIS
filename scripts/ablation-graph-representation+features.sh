#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Ablation study of primal/dual graph representation with similar models on node/edge features
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=ablation/graph_representation+features/dual-gin+gavn,ablation/graph_representation+features/dual-gat+gavn,ablation/graph_representation+features/dual-gin+gavn+pe,ablation/graph_representation+features/dual-gat+gavn+pe,ablation/graph_representation+features/dual-gin+gavn+pe-only,ablation/graph_representation+features/dual-gat+gavn+pe-only,ablation/graph_representation+features/primal-gine+gavn+pe,ablation/graph_representation+features/primal-gat+gavn+pe data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/ablation_graph_representation.log" 2>&1
