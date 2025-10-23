#!/usr/bin/env bash

LOG_DIR=${1:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Ablation study of primal/dual graph representation with similar models on node/edge features
gnn-train -m hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=ablation/graph_representation+features/dual-gin-vcn,ablation/graph_representation+features/dual-gat-vcn,ablation/graph_representation+features/dual-gin-vcn+pe,ablation/graph_representation+features/dual-gat-vcn+pe,ablation/graph_representation+features/dual-gin-vcn+pe-only,ablation/graph_representation+features/dual-gat-vcn+pe-only,ablation/graph_representation+features/primal-gine-vcn+pe,ablation/graph_representation+features/primal-gat-vcn+pe data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/ablation_graph_representation.log" 2>&1
