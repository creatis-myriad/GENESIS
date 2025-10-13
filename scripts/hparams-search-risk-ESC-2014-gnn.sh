#!/usr/bin/env bash

LOG_DIR=${LOG_DIR:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# Use a dedicated hparams search config for GraphGPS, because it has different hyperparameters than the basic GNNs
# NOTE: The number of parallel jobs is reduced to 4 to avoid CUDA out-of-memory errors, since GraphGPS is more memory-intensive
gnn-train logger=wandb trainer=gpu hydra/launcher=joblib hydra.launcher.n_jobs=4 hparams_search=gps_risk_ESC-2014 >>"${LOG_DIR}/hparams_search_graph_gps_risk_ESC-2014.log" 2>&1

# Use the same hparams search config for the message passing GNNs (GCN, GAT, GIN and its variants), but optimize each model separately
for experiment in gcn gat gin gin+vn; do
  gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb trainer=gpu hparams_search=basic_gnn_risk_ESC-2014 experiment=persevere/$experiment >>"${LOG_DIR}/hparams_search_${experiment}_risk_ESC-2014.log" 2>&1
done
