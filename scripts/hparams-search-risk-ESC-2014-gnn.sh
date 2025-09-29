#!/usr/bin/env bash

export LOG_DIR=$1

# Use the same hparams search config for the message passing GNNs (GCN, GIN, GAT), but optimize each model separately
for model in gcn gin gat; do
  gnn-train logger=wandb trainer=gpu hydra/launcher=joblib hydra.launcher.n_jobs=10 hparams_search=basic_gnn_risk_ESC-2014 model/encoder=$model >>"${LOG_DIR}/hparams_search_${model}_risk_ESC-2014.log" 2>&1
done
