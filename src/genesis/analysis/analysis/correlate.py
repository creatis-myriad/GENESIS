import logging
from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots

from genesis.analysis.scores.mastora import mastora
from genesis.analysis.scores.qanadli import qanadli
from genesis.data.utils import find_graph_file, json_to_networkx

log = logging.getLogger(__name__)


def compute_global_obstruction_scores(
    patient_ids: list[str],
    score: Literal["qanadli", "mastora"],
    graphs_dirs: list[Path],
    pattern: str,
    legacy_networkx_format: bool,
    obstruction_attrs: list[str],
) -> pd.DataFrame:
    """Compute scores for each patient in the clinical data."""

    def _compute(graph: nx.DiGraph, attr: str) -> float:
        match score:
            case "qanadli":
                return qanadli(graph, obstruction_attr=attr)
            case "mastora":
                return mastora(graph, obstruction_attr=attr)
            case _:
                raise ValueError(f"Unknown score: {score}")

    obstruction_records = []

    for attr in obstruction_attrs:
        for patient_id in patient_ids:
            try:
                graph_file = find_graph_file(patient_id, search_dirs=graphs_dirs, pattern=pattern)
            except FileNotFoundError:
                continue  # Skip patient if no associated vascular tree graph is found
            except RuntimeError:
                # Log and skip if multiple files are found
                log.exception("", exc_info=True)
                continue

            try:
                graph: nx.DiGraph = json_to_networkx(graph_file, edges="links" if legacy_networkx_format else "edges")
                obstruction_score = _compute(graph, attr)
                obstruction_records.append(
                    {"patient_id": patient_id, "score": obstruction_score, "obstruction_attr": attr}
                )
            except Exception as e:
                log.exception(
                    f"Error processing graph for patient {patient_id} with attr {attr}: {e}",
                    exc_info=True,
                )

    if not obstruction_records:
        raise AssertionError(f"No graphs could be processed from directories: {graphs_dirs}.")

    return pd.DataFrame(obstruction_records)


def plot_correlation(
    data: pd.DataFrame,
    score: str,
    target_attribute: str,
    clinical_data_path: Path,
    graphs_dirs: list[Path],
    cli_command: str,
    show_visualization: bool = False,
) -> None:
    """Plot the correlation using Plotly."""
    pio.renderers.default = "browser"

    obstruction_attrs = data["obstruction_attr"].unique()

    fig = make_subplots(
        rows=1,
        cols=len(obstruction_attrs),
        # Convert snake-case attribute names to title case for better readability
        subplot_titles=[attr.replace("_", " ").title() for attr in obstruction_attrs],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    # Add scatter plots for each attribute
    for i, attr in enumerate(obstruction_attrs):
        attr_data = data[data["obstruction_attr"] == attr]
        corr = attr_data["score"].corr(attr_data[target_attribute])

        log.info(f"Pearson correlation for {attr}: " + (f"{corr:.3f}" if not pd.isna(corr) else "insufficient data"))

        fig.add_scatter(
            x=attr_data["score"],
            y=attr_data[target_attribute],
            mode="markers",
            name=attr,
            marker={"size": 8},
            row=1,
            col=i + 1,
            customdata=attr_data["patient_id"],
            hovertemplate="<b>Patient ID:</b> %{customdata}<br><b>Score:</b> %{x}<br><b>"
            + target_attribute.capitalize()
            + ":</b> %{y}<extra></extra>",
        )

    title_text = (
        f"Correlation between {score.capitalize()} Score and {target_attribute.capitalize()}<br>"
        f"<sup>Clinical Data: <span style='color:blue;'>{clinical_data_path!s}</span> "
        f"| Graphs Directory: <span style='color:blue;'>{', '.join(str(d) for d in graphs_dirs)}</span>"
        f"<br>CLI Command: <span style='color:green;'>{cli_command}</span></sup>"
    )
    fig.update_layout(
        title=title_text,
        title_font_size=18,
        plot_bgcolor="white",
        showlegend=False,
        height=600,
        margin={"t": 150},
    )

    for i in range(len(obstruction_attrs)):
        fig.update_xaxes(
            title_text=f"{score.capitalize()} Score", showgrid=True, gridcolor="lightgray", row=1, col=i + 1
        )
        if i == 0:  # On the first column, set the y-axis title
            fig.update_yaxes(
                title_text=target_attribute.capitalize(), showgrid=True, gridcolor="lightgray", row=1, col=1
            )
        else:
            fig.update_yaxes(showgrid=True, gridcolor="lightgray", row=1, col=i + 1)

    if show_visualization:
        fig.show(renderer="browser")
        log.info("Correlation plot displayed.")
    else:
        log.info("Correlation plot not displayed. Use --show-visualization to view plot.")


def correlate_and_plot(
    score: Literal["qanadli", "mastora"],
    target_attribute: str,
    clinical_data_path: Path,
    graphs_dirs: list[Path],
    pattern: str,
    legacy_networkx_format: bool,
    obstruction_attrs: list[str],
    cli_command: str,
    show_visualization: bool = False,
) -> None:
    """Load data, compute scores, and plot the correlation."""
    log.info(f"Loading clinical data from {clinical_data_path}...")
    data = _load_and_clean_clinical_data(clinical_data_path, target_attribute)

    log.info(f"Compute {score} scores from {obstruction_attrs} attributes...")
    patient_ids = data["patient_id"].tolist()
    scores = compute_global_obstruction_scores(
        patient_ids, score, graphs_dirs, pattern, legacy_networkx_format, obstruction_attrs
    )
    data = pd.merge(data, scores, on=["patient_id"])

    if data.empty:
        raise AssertionError("No data available after merging clinical data and global obstruction score.")

    log.info("Generating correlation plot...")
    plot_correlation(
        data,
        score,
        target_attribute,
        clinical_data_path,
        graphs_dirs,
        cli_command,
        show_visualization=show_visualization,
    )


def _load_and_clean_clinical_data(file_path: Path, target_attribute: str) -> pd.DataFrame:
    """Load and clean clinical data from a CSV file."""
    df = pd.read_csv(file_path, na_values=["nr", "NR"])
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)
    df[target_attribute] = (
        df[target_attribute].astype(str).str.replace("<", "").str.strip().pipe(pd.to_numeric, errors="coerce")
    )
    df.dropna(subset=[target_attribute], inplace=True)
    return df
