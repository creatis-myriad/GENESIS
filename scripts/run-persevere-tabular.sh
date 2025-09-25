#!/usr/bin/env bash

export LOG_DIR=$1

# NOTE: Ignore conflicts on data splits (+data.on_conflict=ignore), because splits computed from different targets would
# not match. This way, the splits computed from the first target will be used for all subsequent targets.

# Multi-class classification risk prediction
baseline-tabular -m logger=wandb test=True data/split=k_fold data/dataset/target=risk_ESC-2014 experiment=tabular_baseline/spesi,tabular_baseline/spesi+cardiac_biomarkers,tabular_baseline/spesi+graph_biomarkers,tabular_baseline/spesi+cardiac_biomarkers+graph_biomarkers +data.on_conflict=ignore model/metrics=multi_classification model/components@model.model=tabpfn,xgboost 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-tabular-multi_classification.log" 2>&1

# Binary classification elevated bio-markers/risk prediction
baseline-tabular -m logger=wandb test=True data/split=k_fold data/dataset/target=risk_ESC-2014_elevated,troponin_elevated,nt-probnp_elevated,enzymes_elevated experiment=tabular_baseline/spesi,tabular_baseline/spesi+cardiac_biomarkers,tabular_baseline/spesi+graph_biomarkers,tabular_baseline/spesi+cardiac_biomarkers+graph_biomarkers +data.on_conflict=ignore model/metrics=binary_classification model/components@model.model=tabpfn,xgboost 'data.split_idx=range(10)' >>"${LOG_DIR}/persevere-tabular-binary_classification.log" 2>&1
