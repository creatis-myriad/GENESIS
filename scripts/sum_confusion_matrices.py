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
    "--labels", type=str, nargs="*", help="Class names to use to label the axes of the confusion matrix"
)
parser.add_argument(
    "--fontsize",
    type=str,
    default="medium",
    help="Font size to use for annotations in the confusion matrix cells. "
    "See https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_fontsize",
)
parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use for the confusion matrix plot")
parser.add_argument(
    "--colorbar", action="store_true", help="Whether to display a colorbar alongside the confusion matrix"
)

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
plt.setp(disp.ax_.get_yticklabels(), rotation=45)
plt.tight_layout()
if args.output_file:
    plt.savefig(args.output_file)
else:
    plt.show()
