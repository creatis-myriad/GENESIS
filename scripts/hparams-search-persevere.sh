#!/usr/bin/env bash

TARGET=${1:-"enzymes_elevated"}  # Default to "enzymes_elevated" if TARGET is not set by the user
LOG_DIR=${2:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, pre-computed splits will be used if available, otherwise splits will be computed for the target.

# Use the same hparams search config for the message passing GNNs (GCN, GAT, GIN and its virtual node variants), but optimize each model separately
for model in gcn gat gin gin+vn gin-vcn; do
  gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=10 logger=wandb trainer=gpu hparams_search=persevere_basic_gnn experiment=persevere/"${TARGET}"/$model +data.on_conflict=ignore >>"${LOG_DIR}/hparams_search_${model}_${TARGET}.log" 2>&1
done

# Use a dedicated hparams search config for GPS, because it has different hyperparameters than the basic GNNs
# NOTE: The number of parallel jobs is reduced to 4 to avoid CUDA out-of-memory errors, since GPS is more memory-intensive
gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=4 logger=wandb trainer=gpu hparams_search=persevere_gps experiment=persevere/"${TARGET}"/gps +data.on_conflict=ignore model/metrics=binary_classification >>"${LOG_DIR}/hparams_search_gps_${TARGET}.log" 2>&1
