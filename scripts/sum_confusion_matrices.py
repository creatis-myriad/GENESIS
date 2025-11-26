import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=Path, help="Directory containing confusion matrix as CSV files")
parser.add_argument("--output_file", type=Path, help="Path to save the aggregated confusion matrix as an image")
parser.add_argument(
    "--labels",
    type=str,
    nargs="*",
    help="Class names to use to label the axes of the confusion matrix. If not provided, integer indices will be used.",
)
parser.add_argument(
    "--true_title", type=str, help="Title for the true label axis. If not provided, defaults to 'True label'."
)
parser.add_argument(
    "--predicted_title",
    type=str,
    help="Title for the predicted label axis. If not provided, defaults to 'Predicted label'.",
)
parser.add_argument(
    "--fontsize",
    type=str,
    default="medium",
    help="Font size to use for labels and annotations in the confusion matrix cells. "
    "See https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_fontsize",
)
parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use for the confusion matrix plot")
parser.add_argument(
    "--colorbar", action="store_true", help="Whether to display a colorbar alongside the confusion matrix"
)
parser.add_argument("--ytick_rotation", type=int, help="Rotation for y-axis tick labels")
parser.add_argument("--xtick_rotation", type=int, help="Rotation for x-axis tick labels")

args = parser.parse_args()

# Read all CSV files in the input directory as confusion matrices
# NOTE 1: CSV files are expected to have no header or index column
# NOTE 2: Rows are expected to correspond to true classes, columns to predicted classes
files = list(args.input_dir.glob("*.csv"))
print(f"Found {len(files)} CSV files in {args.input_dir}: {[file.name for file in files]}")
confusion_matrices = [pd.read_csv(file, header=None).to_numpy() for file in files]

# Aggregate confusion matrices by summing them
agg_confusion_matrix = np.sum(confusion_matrices, axis=0)

# Plot the aggregated confusion matrix using sklearn's ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=agg_confusion_matrix, display_labels=args.labels)
disp = disp.plot(cmap=args.cmap, text_kw={"fontsize": args.fontsize}, colorbar=args.colorbar)
# Adjust axis labels
if args.true_title:
    disp.ax_.set_ylabel(args.true_title)
if args.predicted_title:
    disp.ax_.set_xlabel(args.predicted_title)
# Adjust axis/ticks' labels font sizes
disp.ax_.xaxis.label.set_size(args.fontsize)
disp.ax_.yaxis.label.set_size(args.fontsize)
disp.ax_.tick_params(axis="both", labelsize=args.fontsize)
# Adjust tick labels rotations if specified
if args.ytick_rotation:
    plt.setp(disp.ax_.get_yticklabels(), rotation=args.ytick_rotation)
if args.xtick_rotation:
    plt.setp(disp.ax_.get_xticklabels(), rotation=args.xtick_rotation)
if args.output_file:
    plt.savefig(args.output_file, bbox_inches="tight")
else:
    plt.tight_layout()
    plt.show()
