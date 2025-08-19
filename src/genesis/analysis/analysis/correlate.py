from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from genesis.analysis.scores.mastora import compute_mastora
from genesis.analysis.scores.qanadli import compute_qanadli
from genesis.data.utils import json_to_networkx
from genesis.utils import RankedLogger

from ..utils import find_graph_file  # noqa: TID252

log = RankedLogger(__name__, rank_zero_only=True)


def load_and_clean_clinical_data(file_path: Path, attribute: str) -> pd.DataFrame:
    """Load and clean clinical data from a CSV file."""
    df = pd.read_csv(file_path)
    df[attribute] = df[attribute].astype(str).str.replace("<", "").str.strip().pipe(pd.to_numeric, errors="coerce")
    df.dropna(subset=[attribute], inplace=True)
    return df


def calculate_scores(
    score_name: str,
    clinical_data: pd.DataFrame,
    graphs_dirs: list[Path],
    obstruction_attr: str,
    all_attributes: bool = False,
) -> pd.DataFrame:
    """Calculate scores for each patient in the clinical data."""

    def _compute(graph: nx.Graph, attr: str) -> float:
        if score_name == "mastora":
            return compute_mastora(graph, obstruction_attr=attr)
        return compute_qanadli(graph, obstruction_attr=attr)

    records = []
    attrs = (
        [
            "max_transversal_obstruction",
            "max_transversal_obstruction_propagated",
            "max_transversal_obstruction_cumulated",
        ]
        if all_attributes
        else [obstruction_attr]
    )

    for attr in attrs:
        for _, row in clinical_data.iterrows():
            pid = str(row["patient_id"]).zfill(4)
            try:
                graph_file = find_graph_file(
                    Path(pid),
                    search_dirs=graphs_dirs,
                    pattern=f"*{pid}*.json",
                )
            except FileNotFoundError:
                continue

            try:
                graph = json_to_networkx(graph_file)
                score = _compute(graph, attr)
                rec = {"patient_id": row["patient_id"], "score": score}
                if all_attributes:
                    rec["obstruction_attr"] = attr
                records.append(rec)
            except Exception as e:
                log.exception(
                    f"Error processing graph for patient {pid} with attr {attr}: {e}",
                    exc_info=True,
                )

    df_scores = pd.DataFrame(records)
    if all_attributes:
        # For all_attributes, we need to create multiple rows per patient (one per obstruction_attr)
        expanded_clinical = []
        for attr in attrs:
            temp_df = clinical_data.copy()
            temp_df["obstruction_attr"] = attr
            expanded_clinical.append(temp_df)
        expanded_clinical_data = pd.concat(expanded_clinical, ignore_index=True)
        return pd.merge(expanded_clinical_data, df_scores, on=["patient_id", "obstruction_attr"])
    return pd.merge(clinical_data, df_scores, on=["patient_id"])


def plot_correlation(
    data: pd.DataFrame,
    score_name: str,
    attribute: str,
    clinical_data_path: str,
    graphs_dirs: list[Path],
    obstruction_attr: str,
    cli_command: str,
    all_attributes: bool = False,
    show_visualization: bool = False,
) -> None:
    """Plot the correlation using Plotly."""
    pio.renderers.default = "browser"

    if all_attributes:
        unique_attrs = data["obstruction_attr"].unique()
        n_attrs = len(unique_attrs)

        fig = make_subplots(
            rows=1,
            cols=n_attrs,
            subplot_titles=[
                attr.replace("max_transversal_obstruction", "Max Transversal Obstruction")
                .replace("_propagated", " Propagated")
                .replace("_cumulated", " Cumulated")
                for attr in unique_attrs
            ],
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        # Add scatter plots for each attribute
        for i, attr in enumerate(unique_attrs):
            attr_data = data[data["obstruction_attr"] == attr]
            corr = attr_data["score"].corr(attr_data[attribute])

            log.info(
                f"Pearson correlation for {attr}: " + (f"{corr:.3f}" if not pd.isna(corr) else "insufficient data")
            )

            fig.add_scatter(
                x=attr_data["score"],
                y=attr_data[attribute],
                mode="markers",
                name=attr,
                marker={"size": 8},
                row=1,
                col=i + 1,
                customdata=attr_data["patient_id"],
                hovertemplate="<b>Patient ID:</b> %{customdata}<br><b>Score:</b> %{x}<br><b>"
                + attribute.capitalize()
                + ":</b> %{y}<extra></extra>",
            )

        title_text = (
            f"Correlation between {score_name.capitalize()} Score and {attribute.capitalize()}<br>"
            f"<sup>Clinical Data: <span style='color:blue;'>{clinical_data_path}</span> "
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

        for i in range(n_attrs):
            fig.update_xaxes(
                title_text=f"{score_name.capitalize()} Score", showgrid=True, gridcolor="lightgray", row=1, col=i + 1
            )
            if i == 0:
                fig.update_yaxes(title_text=attribute.capitalize(), showgrid=True, gridcolor="lightgray", row=1, col=1)
            else:
                fig.update_yaxes(showgrid=True, gridcolor="lightgray", row=1, col=i + 1)

    else:
        corr = data["score"].corr(data[attribute])
        log.info("Pearson correlation: " + (f"{corr:.3f}" if not pd.isna(corr) else "insufficient data"))

        title_text = (
            f"Correlation between {score_name.capitalize()} Score and {attribute.capitalize()}<br>"
            f"<sup>Clinical Data: <span style='color:blue;'>{clinical_data_path}</span> "
            f"| Graphs Directory: <span style='color:blue;'>{', '.join(str(d) for d in graphs_dirs)}</span> "
            f"| Obstruction Attribute: <span style='color:blue;'>{obstruction_attr}</span>"
            f"<br>CLI Command: <span style='color:green;'>{cli_command}</span></sup>"
        )
        fig = px.scatter(
            data,
            x="score",
            y=attribute,
            title=title_text,
            labels={"score": f"{score_name.capitalize()} Score", attribute: attribute.capitalize()},
            hover_data=["patient_id"],
        )
        fig.update_traces(marker={"size": 20})
        fig.update_layout(
            plot_bgcolor="white",
            xaxis={"showgrid": True, "gridcolor": "lightgray"},
            yaxis={"showgrid": True, "gridcolor": "lightgray"},
            title_font_size=24,
        )

    if show_visualization:
        fig.show(renderer="browser")
        log.info("Correlation plot displayed.")
    else:
        log.info("Correlation plot not displayed. Use --show-visualization to view plot.")


def correlate_and_plot(
    score_name: str,
    attribute: str,
    clinical_data_path: Path,
    graphs_dirs: list[Path],
    obstruction_attr: str,
    cli_command: str,
    all_attributes: bool = False,
    show_visualization: bool = False,
) -> None:
    """Load data, calculate scores, and plot the correlation."""
    log.info(f"Loading clinical data from {clinical_data_path}...")
    clinical_df = load_and_clean_clinical_data(clinical_data_path, attribute)

    log.info(
        f"Calculating {score_name} scores "
        + ("for all obstruction attributes..." if all_attributes else "for patients...")
    )

    data_with_scores = calculate_scores(
        score_name,
        clinical_df,
        graphs_dirs,
        obstruction_attr,
        all_attributes,
    )

    if data_with_scores.empty:
        log.info("No data to plot. Make sure graph files exist and patient IDs match.", err=True)
        return

    if show_visualization:
        log.info("Generating correlation plot...")
    plot_correlation(
        data_with_scores,
        score_name,
        attribute,
        str(clinical_data_path),
        graphs_dirs,
        obstruction_attr,
        cli_command,
        all_attributes,
        show_visualization,
    )
