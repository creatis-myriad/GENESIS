#!/usr/bin/env bash

LOG_DIR=${LOG_DIR:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Ablation study of primal/dual graph representation with models as similar as possible (GIN/GINE w/ positional encoding)
# that handle the swapped node/edge features
gnn-train -m logger=wandb logger.wandb.log_model=True trainer=gpu test=True experiment=persevere/dual-gin+pe,persevere/primal-gine+pe data/split=k_fold +data.on_conflict=ignore 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-gnn-primal_dual_ablation.log" 2>&1
