import itertools
import logging
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

_CATEGORICAL_CLINICAL_ATTRS = {
    "risk": [0, 1, 2],
    "spesi": [0, 1, 2],
    "cancer_history": [0, 1],
    "cpd_history": [0, 1],
    "inverted_rv-lv_ratio": [0, 1],
}
_LABELS = {
    "patient_id": "Patient ID",
    "mastora_central": "Mastora (central) score",
    "mastora_peripheral": "Mastora (peripheral) score",
    "mastora_global": "Mastora (global) score",
    "qanadli": "Qanadli score",
    "risk": "ESC Guidelines risk level",
    "spesi": "SPESI Score",
    "bnp": "BNP",
    "troponin": "Troponin",
}


def plot_correlation(
    data: pd.DataFrame,
    scores: list[str],
    clinical_attrs: list[str],
    cli_command: str,
    categorical_plot: Literal["violin", "histogram"] = "violin",
) -> None:
    """Plot the correlation between scores computed on vascular graph and clinical attributes using Plotly.

    Args:
        data: DataFrame containing patient clinical data and computed vascular graph scores.
        scores: Score(s) computed on the vascular graph (e.g., "qanadli") to correlate with `clinical_attrs`.
        clinical_attrs: Clinical attribute(s) (e.g., "risk", "troponin") to correlate with `scores`.
        cli_command: CLI command used to generate the data, to display in the plot title for reproducibility.
        categorical_plot: Type of plot to use for categorical clinical attributes.
    """
    if (missing_scores := pd.Index(scores).difference(data.columns)).any():
        raise ValueError(f"The following requested scores are missing from the data: {missing_scores.tolist()}.")
    if (missing_attrs := pd.Index(clinical_attrs).difference(data.columns)).any():
        raise ValueError(
            f"The following requested clinical attributes are missing from the data: {missing_attrs.tolist()}."
        )

    categorical_attrs = [attr for attr in clinical_attrs if attr in _CATEGORICAL_CLINICAL_ATTRS]
    if len(categorical_attrs) > 1:
        raise ValueError(
            f"Because of limitations with Plotly legends, only one categorical clinical attribute can be plotted at a "
            f"time, but multiple were requested: {categorical_attrs}. "
            f"Please run the function multiple times with one categorical attribute at a time."
        )

    data = data.reset_index(names="patient_id")  # Set patient_id index as column to plot it easily

    fig = make_subplots(
        rows=len(clinical_attrs),
        cols=len(scores),
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[
            f"{_LABELS.get(attr, attr)} vs {_LABELS.get(score, score)}"
            for attr, score in itertools.product(clinical_attrs, scores)
        ]
        # Only display individual subplot titles if there are more than one row and column
        if len(clinical_attrs) * len(scores) > 1
        else None,
    )

    # Add plots for each combination of clinical attribute and graph score
    for (i, clinical_attr), (j, score) in itertools.product(enumerate(clinical_attrs), enumerate(scores)):
        corr = data[clinical_attr].corr(data[score])

        log.info(
            f"Pearson correlation between {clinical_attr} and {score}: "
            + (f"{corr:.3f}" if not pd.isna(corr) else "insufficient data")
        )

        clinical_attr_lbl = _LABELS.get(clinical_attr, clinical_attr)
        score_lbl = _LABELS.get(score, score)

        if clinical_attr in _CATEGORICAL_CLINICAL_ATTRS:  # For categorical clinical attributes, use a 2D histogram
            for cat_idx, cat in enumerate(_CATEGORICAL_CLINICAL_ATTRS[clinical_attr]):
                # Add a histogram w.r.t. the score for each category of the clinical attribute
                cat_data = data[data[clinical_attr] == cat]
                dist_kwargs = {
                    "name": str(cat),
                    "legendgroup": f"{clinical_attr}={cat}",
                    "showlegend": not bool(j),  # only show legend in 1st col, to avoid duplicate entries in the group
                    # explicitly map category to specific color to avoid Plotly using different colors across subplots
                    "marker_color": px.colors.qualitative.Plotly[cat_idx],
                }
                match categorical_plot:
                    case "histogram":
                        dist = go.Histogram(x=cat_data[score], **dist_kwargs)
                    case "violin":
                        dist = go.Violin(
                            x=cat_data[score],
                            y=cat_data[clinical_attr],
                            orientation="h",
                            box_visible=True,  # Show box plot inside the violin
                            meanline_visible=True,  # Show mean line inside the violin
                            spanmode="hard",  # Do not extend the violin beyond the data range
                            **dist_kwargs,
                        )
                    case _:
                        raise ValueError(f"Invalid `categorical_plot` value '{categorical_plot}'.")

                fig.add_trace(dist, row=i + 1, col=j + 1)

            # Title legend according to the clinical attribute
            fig.update_layout(legend_title_text=clinical_attr_lbl)

        else:  # For continuous clinical attributes, use a scatter plot
            scatter = go.Scatter(
                x=data[score],
                y=data[clinical_attr],
                mode="markers",
                customdata=data["patient_id"],
                hovertemplate=f"<b>Patient ID:</b> %{{customdata}}"
                f"<br><b>{score_lbl}:</b> %{{x}}"
                f"<br><b>{clinical_attr_lbl}:</b> %{{y}}"
                f"<extra></extra>",
                showlegend=False,
            )
            fig.add_trace(scatter, row=i + 1, col=j + 1)

    # Set axes titles on the left and bottom of the grid
    for i, clinical_attr in enumerate(clinical_attrs):
        clinical_axis_title = _LABELS.get(clinical_attr, clinical_attr)
        if clinical_attr in _CATEGORICAL_CLINICAL_ATTRS and categorical_plot == "histogram":
            # For histogram plots, indicate the y-axis is a count
            clinical_axis_title = "Patient count"
        fig.update_yaxes(title_text=clinical_axis_title, row=i + 1, col=1)

    for j, score in enumerate(scores):
        fig.update_xaxes(title_text=_LABELS.get(score, score), row=len(clinical_attrs), col=j + 1)

    title_text = (
        f"Correlations between vascular scores and clinical parameters<br>"
        f"<sup><br>"
        f"CLI command: <span style='color:green;'>{cli_command}</span>"
        f"</sup>"
    )
    fig.update_layout(
        template="plotly",
        title={
            "text": title_text,
            "font_size": 18,
        },
        margin={"t": 150},
    )
    fig.show(renderer="browser")
