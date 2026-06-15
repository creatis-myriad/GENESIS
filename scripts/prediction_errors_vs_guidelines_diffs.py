import collections
import itertools
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

from genesis.data.utils.io import load_and_clean_tabular_features

TARGET_FEATURES = ["risk_ESC-2014", "risk_ESC-2019"]


def main() -> None:
    """Main entry point for extracting PERSEVERE features by specified subsets."""
    parser = ArgumentParser(
        description="Compare errors in risk predictions vs differences in target stratification between guidelines."
    )
    parser.add_argument(
        "predictions_root", type=Path, help="Path to the root directory containing prediction CSV files."
    )
    parser.add_argument("splits_file", type=Path, help="Path to the JSON file defining data splits.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        type=str,
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Subset to extract features for, each separately. If not provided, all subsets are extracted.",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        help="Name to use for the stratification model when labeling the confusion matrix axes.",
    )
    parser.add_argument(
        "--fontsize",
        type=str,
        default="xx-large",
        help="Font size to use for labels and annotations in the confusion matrix cells. "
        "See https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_fontsize",
    )
    parser.add_argument("--output_dir", type=Path, help="Directory where to save the confusion matrix as an image.")
    parser.add_argument(
        "--cm_format",
        type=str,
        choices=["svg", "pdf"],
        default="pdf",
        help="Format to save the confusion matrix image.",
    )
    args = parser.parse_args()

    data = load_and_clean_tabular_features(
        rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/global_features.csv"
    )
    data = data[TARGET_FEATURES]

    with args.splits_file.open() as f:
        splits = json.load(f)

    target_diffs = collections.defaultdict(list)
    pred_errors = collections.defaultdict(list)
    f1s = collections.defaultdict(lambda: collections.defaultdict(list))
    for (idx, split), subset in itertools.product(enumerate(splits), args.subsets):
        # Get data for the current subset, making sure to use numerical indices and not patient indices
        indices = split[subset]
        data_subset = data.iloc[indices]

        # Load predictions from corresponding CSV file
        predictions_file = args.predictions_root / str(idx) / "predictions" / f"{subset}_argmax_predictions.csv"
        predictions_df = pd.read_csv(predictions_file, index_col="batch_idx")

        # Merge predictions with the data subset
        # Use the raw prediction values to drop their numerical index inside the subset
        # in favor of the patient index from the target data
        data_subset = data_subset.assign(risk_prediction=predictions_df["prediction"].values)

        # Compute changes in risk stratification between guideline versions
        target_diff = data_subset[TARGET_FEATURES[0]] != data_subset[TARGET_FEATURES[1]]

        # Compute errors in predictions compared to 1st guideline version
        pred_error = data_subset["risk_prediction"] != data_subset[TARGET_FEATURES[0]]

        # Compute prediction F1-scores for each guideline version,
        # as a sanity check that they match reported results
        for risk_feature in TARGET_FEATURES:
            f1 = f1_score(
                data_subset[risk_feature],
                data_subset["risk_prediction"],
                average="macro",
            )
            f1s[risk_feature][subset].append(f1)

        target_diffs[subset].append(target_diff)
        pred_errors[subset].append(pred_error)

    # Print F1-scores for each guideline version and subset
    for risk_feature in TARGET_FEATURES:
        print(f"F1-score vs {risk_feature} over subsets:")
        for subset in args.subsets:
            f1s_mean = np.mean(f1s[risk_feature][subset])
            f1s_std = np.std(f1s[risk_feature][subset])
            print(f"  {subset}: {f1s_mean:.3f} ± {f1s_std:.3f}")

    # Create confusion matrices comparing changes in risk stratification vs prediction errors
    for subset in args.subsets:
        subset_target_diffs = np.hstack(target_diffs[subset])
        subset_pred_errors = np.hstack(pred_errors[subset])
        cm = confusion_matrix(subset_target_diffs, subset_pred_errors)

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(cmap="Blues", text_kw={"fontsize": args.fontsize}, colorbar=False)
        # Adjust axis labels
        disp.ax_.set_ylabel("Different stratification using \n 2014 vs 2019 guidelines")
        xlabel = "Prediction error"
        if args.model_label:
            xlabel = f"{args.model_label} \n prediction error"
        disp.ax_.set_xlabel(xlabel)
        # Adjust axis/ticks' labels font sizes
        disp.ax_.xaxis.label.set_size(args.fontsize)
        disp.ax_.yaxis.label.set_size(args.fontsize)
        disp.ax_.tick_params(axis="both", labelsize=args.fontsize)

        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{subset}_confusion_matrix.{args.cm_format}"
            if args.model_label:
                output_filename = f"{args.model_label.lower()}_{output_filename}"
            cm_filepath = args.output_dir / output_filename
            print(f"Saving confusion matrix for subset '{subset}' to '{cm_filepath}'.")
            plt.savefig(cm_filepath, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

        # Make sure to close all figures between subsets
        plt.close("all")


if __name__ == "__main__":
    main()
