#!/usr/bin/env bash

TARGET=${1:-"enzymes_elevated"}  # Default to "enzymes_elevated" if TARGET is not set by the user
LOG_DIR=${2:-./logs}  # Default to "logs" if LOG_DIR is not set by the user

# NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, pre-computed splits will be used if available, otherwise splits will be computed for the target.

# Use the same hparams search config for the message passing GNNs (GCN, GAT, GIN) and their variants, but optimize each configuration separately
for model in gcn gat gin; do
  for variant in "" +vn +gaef +galf +gavn; do
    gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu logger=wandb \
      hparams_search=persevere_basic_gnn experiment=persevere/"${TARGET}"/${model}${variant} \
      +data.on_conflict=ignore >>"${LOG_DIR}/hparams_search_${model}${variant}_${TARGET}.log" 2>&1
  done

  # Use a dedicated hparams search config for VN_G variants, because they have different hyperparameters than the other variants
  for variant in +vn_g +vn_gv2; do
    gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu logger=wandb \
      hparams_search=persevere_vn_g experiment=persevere/"${TARGET}"/${model}${variant} \
      +data.on_conflict=ignore >>"${LOG_DIR}/hparams_search_${model}${variant}_${TARGET}.log" 2>&1
  done
done

# Use a dedicated hparams search config for GPS and its variants, because they have different hyperparameters than the basic GNNs
for model in gps gagps; do
  gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu logger=wandb \
    hparams_search=persevere_gps experiment=persevere/"${TARGET}"/${model} \
    +data.on_conflict=ignore >>"${LOG_DIR}/hparams_search_${model}_${TARGET}.log" 2>&1
done

# Separate loop for MLP baseline, since its variants are different from other GNNs
model=mlp
for variant in "" +gaef +galf; do
  gnn-train hydra/launcher=joblib hydra.launcher.n_jobs=10 trainer=gpu logger=wandb \
    hparams_search=persevere_basic_gnn experiment=persevere/"${TARGET}"/${model}${variant} \
    +data.on_conflict=ignore >>"${LOG_DIR}/hparams_search_${model}${variant}_${TARGET}.log" 2>&1
done
