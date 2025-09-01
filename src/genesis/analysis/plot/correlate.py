import itertools
import logging

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

_CATEGORICAL_CLINICAL_ATTRS = {
    "risk",
    "spesi",
    "cancer_history",
    "cpd_history",
    "inverted_rv-lv_ratio",
}
_LABELS = {
    "patient_id": "Patient ID",
    "mastora_proximal": "Mastora (proximal) score",
    "mastora_distal": "Mastora (distal) score",
    "qanadli": "Qanadli score",
    "risk": "ESC Guidelines risk Level",
    "spesi": "SPESI Score",
    "bnp": "BNP",
    "troponin": "Troponin",
}


def plot_correlation(
    data: pd.DataFrame,
    scores: list[str],
    clinical_attrs: list[str],
    cli_command: str,
) -> None:
    """Plot the correlation between scores computed on vascular graph and clinical attributes using Plotly.

    Args:
        data: DataFrame containing patient clinical data and computed vascular graph scores.
        scores: Score(s) computed on the vascular graph (e.g., "qanadli") to correlate with `clinical_attrs`.
        clinical_attrs: Clinical attribute(s) (e.g., "risk", "troponin") to correlate with `scores`.
        cli_command: CLI command used to generate the data, to display in the plot title for reproducibility.
    """
    if (missing_scores := pd.Index(scores).difference(data.columns)).any():
        raise ValueError(f"The following requested scores are missing from the data: {missing_scores.tolist()}.")
    if (missing_attrs := pd.Index(clinical_attrs).difference(data.columns)).any():
        raise ValueError(
            f"The following requested clinical attributes are missing from the data: {missing_attrs.tolist()}."
        )

    data = data.reset_index(names="patient_id")  # Set patient_id index as column to plot it easily
    # Update labels for clinical attributes and scores for better readability in the plots
    data = data.rename(columns=_LABELS)
    scores = [_LABELS.get(score, score) for score in scores]
    clinical_attrs = [_LABELS.get(attr, attr) for attr in clinical_attrs]

    fig = make_subplots(
        rows=len(clinical_attrs),
        cols=len(scores),
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[f"{attr} vs {score}" for attr, score in itertools.product(clinical_attrs, scores)]
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

        if clinical_attr in _CATEGORICAL_CLINICAL_ATTRS:
            raise NotImplementedError(
                f"Correlation plots for categorical clinical attributes like '{clinical_attr}' are not implemented yet."
            )
        fig.add_trace(
            go.Scatter(
                x=data[score],
                y=data[clinical_attr],
                mode="markers",
                customdata=data["Patient ID"],
                hovertemplate=f"<b>Patient ID:</b> %{{customdata}}"
                f"<br><b>{score}:</b> %{{x}}"
                f"<br><b>{clinical_attr}</b> %{{y}}"
                f"<extra></extra>",
            ),
            row=i + 1,
            col=j + 1,
        )

    # Set axes titles on the left and bottom of the grid
    for i, clinical_attr in enumerate(clinical_attrs):
        fig.update_yaxes(title_text=clinical_attr, row=i + 1, col=1)
    for j, score in enumerate(scores):
        fig.update_xaxes(title_text=score, row=len(clinical_attrs), col=j + 1)

    title_text = (
        f"Correlations between vascular scores and clinical parameters<br>"
        f"<sup>"
        f"CLI Command: <span style='color:green;'>{cli_command}</span>"
        f"</sup>"
    )
    fig.update_layout(
        template="plotly",
        title=title_text,
        title_font_size=18,
        height=600,
        margin={"t": 150},
    )
    fig.show(renderer="browser")
