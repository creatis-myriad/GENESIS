import itertools
import logging
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from genesis.analysis.config import (
    PERSEVERE_ATTRS_CATEGORIES,
    PERSEVERE_ATTRS_LABELS,
    PERSEVERE_ATTRS_RANGES,
)

log = logging.getLogger(__name__)


def facet_grid(
    data: pd.DataFrame,
    col_attrs: list[str],
    row_attrs: list[str],
    cli_command: str,
    categorical_plot: Literal["violin", "histogram"] = "violin",
) -> None:
    """Plot the correlation between input clinical attributes and/or measured scores using Plotly.

    Args:
        data: DataFrame containing patient clinical data and computed vascular graph scores.
        col_attrs: Attributes (e.g. 'risk', 'qanadli', etc.) to plot on the columns of the grid.
        row_attrs: Attributes (e.g. 'risk', 'qanadli', etc.) to plot on the rows of the grid.
        cli_command: CLI command used to generate the data, to display in the plot title for reproducibility.
        categorical_plot: Type of plot to use for categorical clinical attributes.
    """
    if (missing_attrs := pd.Index(col_attrs + row_attrs).difference(data.columns)).any():
        raise ValueError(f"The following requested attributes are missing from the data: {missing_attrs.tolist()}.")

    categorical_attrs = [attr for attr in (col_attrs + row_attrs) if attr in PERSEVERE_ATTRS_CATEGORIES]
    if len(categorical_attrs) > 1:
        raise ValueError(
            f"Because of limitations with Plotly legends, only one categorical attribute can be plotted at a "
            f"time, but multiple were requested: {categorical_attrs}. "
            f"Please run the function multiple times with one categorical attribute at a time."
        )

    data = data.reset_index(names="patient_id")  # Set patient_id index as column to plot it easily

    fig = make_subplots(
        rows=len(row_attrs),
        cols=len(col_attrs),
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[
            f"{PERSEVERE_ATTRS_LABELS.get(row_attr, row_attr)} vs {PERSEVERE_ATTRS_LABELS.get(col_attr, col_attr)}"
            for row_attr, col_attr in itertools.product(row_attrs, col_attrs)
        ]
        # Only display individual subplot titles if there are more than one row and column
        if len(row_attrs) * len(col_attrs) > 1
        else None,
    )

    _add_traces_to_grid(fig, data, row_attrs, col_attrs, categorical_plot)

    # Customize properties of the axes, based on their specific attributes
    _customize_axes_layout(fig, row_attrs, "row", col_attrs, categorical_plot)
    _customize_axes_layout(fig, col_attrs, "col", row_attrs, categorical_plot)

    title_text = (
        f"Distributions between clinical parameters and vascular scores<br>"
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


def _add_traces_to_grid(
    fig: go.Figure,
    data: pd.DataFrame,
    row_attrs: list[str],
    col_attrs: list[str],
    categorical_plot: Literal["violin", "histogram"],
) -> None:
    # Add plots for each combination in the grid
    for (i, row_attr), (j, col_attr) in itertools.product(enumerate(row_attrs), enumerate(col_attrs)):
        corr = data[row_attr].corr(data[col_attr])

        log.info(
            f"Pearson correlation between {row_attr} and {col_attr}: "
            + (f"{corr:.3f}" if not pd.isna(corr) else "insufficient data")
        )

        row_attr_lbl = PERSEVERE_ATTRS_LABELS.get(row_attr, row_attr)
        col_attr_lbl = PERSEVERE_ATTRS_LABELS.get(col_attr, col_attr)

        if (is_row_cat := row_attr in PERSEVERE_ATTRS_CATEGORIES) or (
            col_attr in PERSEVERE_ATTRS_CATEGORIES
        ):  # For categorical clinical attributes, use a 2D histogram
            cat_attr = row_attr if is_row_cat else col_attr

            for cat_idx, cat in enumerate(PERSEVERE_ATTRS_CATEGORIES[cat_attr]):
                # Add a histogram w.r.t. the score for each category of the clinical attribute
                cat_data = data[data[cat_attr] == cat]
                dist_kwargs = {
                    "name": str(cat),
                    "legendgroup": f"{cat_attr}={cat}",
                    "showlegend": not bool(
                        j if is_row_cat else i
                    ),  # only show legend in 1st row/col, to avoid duplicate entries in the group
                    # explicitly map category to specific color to avoid Plotly using different colors across subplots
                    "marker_color": px.colors.qualitative.Plotly[cat_idx],
                }
                match categorical_plot:
                    case "histogram":
                        dist = go.Histogram(
                            x=cat_data[col_attr] if is_row_cat else None,
                            y=None if is_row_cat else cat_data[row_attr],
                            **dist_kwargs,
                        )
                    case "violin":
                        dist = go.Violin(
                            x=cat_data[col_attr],
                            y=cat_data[row_attr],
                            orientation="h" if is_row_cat else "v",
                            box_visible=True,  # Show box plot inside the violin
                            meanline_visible=True,  # Show mean line inside the violin
                            spanmode="hard",  # Do not extend the violin beyond the data range
                            **dist_kwargs,
                        )
                    case _:
                        raise ValueError(f"Invalid `categorical_plot` value '{categorical_plot}'.")

                fig.add_trace(dist, row=i + 1, col=j + 1)

            # Title legend according to the clinical attribute
            fig.update_layout(legend_title_text=row_attr_lbl if is_row_cat else col_attr_lbl)

        else:  # For continuous clinical attributes, use a scatter plot
            scatter = go.Scatter(
                x=data[col_attr],
                y=data[row_attr],
                mode="markers",
                customdata=data["patient_id"],
                hovertemplate=f"<b>Patient ID:</b> %{{customdata}}"
                f"<br><b>{col_attr_lbl}:</b> %{{x}}"
                f"<br><b>{row_attr_lbl}:</b> %{{y}}"
                f"<extra></extra>",
                showlegend=False,
            )
            fig.add_trace(scatter, row=i + 1, col=j + 1)


def _customize_axes_layout(
    fig: go.Figure,
    attrs: list[str],
    axis: Literal["row", "col"],
    opposite_attrs: list[str],
    categorical_plot: Literal["violin", "histogram"],
) -> None:
    for idx, attr in enumerate(attrs):
        update_kwargs = {}

        axis_title = PERSEVERE_ATTRS_LABELS.get(attr, attr)
        if attr in PERSEVERE_ATTRS_CATEGORIES and categorical_plot == "histogram":
            # For histogram plots, indicate the axis is a count
            axis_title = "Patient count"

        # Set axis range manually if defined for the attribute
        elif attr_range := PERSEVERE_ATTRS_RANGES.get(attr):
            opposite_cat_attrs = [attr for attr in opposite_attrs if attr in PERSEVERE_ATTRS_CATEGORIES]
            if any(opposite_cat_attrs) and categorical_plot == "histogram":
                shared_axis = "y" if axis == "row" else "x"
                log.warning(
                    f"Setting pre-defined {shared_axis}-axis range {attr_range} for attribute '{attr}' when histograms "
                    f"are requested for categorical attributes {opposite_cat_attrs} would risk cropping out bars at "
                    f"the edge; ignoring the range. \n"
                    f"To avoid this warning, use violin plots instead of histograms for categorical attributes."
                )
            else:
                update_kwargs["range"] = attr_range

        if axis == "row":
            # Apply updates shared across the y-axes of the row
            fig.update_yaxes(**update_kwargs, row=idx + 1)
            # Set row title on the left (1st column) of the grid
            fig.update_yaxes(title_text=axis_title, row=idx + 1, col=1)
        else:  # axis == "col"
            # Apply updates shared across the x-axes of the column
            fig.update_xaxes(**update_kwargs, col=idx + 1)
            # Set column title on the bottom (last row) of the grid
            fig.update_xaxes(title_text=axis_title, row=len(opposite_attrs), col=idx + 1)
