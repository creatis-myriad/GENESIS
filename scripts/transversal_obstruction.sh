#!/bin/bash

INPUT_DIR=$1
VESSEL_MASKS_DIR=$2
OBSTRUCTION_MASKS_DIR=$3

for graph_filepath in "${INPUT_DIR}"/*.json; do
  echo "Processing $graph_filepath"
  graph_filename=$(basename "$graph_filepath" .json)
  graph_id=${graph_filename::4}
  python src/genesis/data/persevere/transversal_obstruction.py \
    "${VESSEL_MASKS_DIR}/${graph_id}_arteries.nii.gz" \
    "${OBSTRUCTION_MASKS_DIR}/${graph_id}_embolism.nii.gz" \
    "${graph_filepath}"
done
