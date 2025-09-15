import os
import sys
from pathlib import Path

import click
import networkx as nx
import pandas as pd
import rootutils
from tqdm.auto import tqdm

from genesis.analysis.cli.commands import mastora, qanadli, visualize
from genesis.analysis.cli.parameters import patient_data_params
from genesis.analysis.cli.utils import get_logger
from genesis.analysis.config import PERSEVERE_ATTRS_LABELS, PERSEVERE_AUTO_MEASURES
from genesis.analysis.plot.distribution import facet_grid
from genesis.data.utils.io import find_file, json_to_networkx, load_and_clean_clinical_data

log = get_logger(__name__)


@click.group("eval-graph", chain=True)
@click.option(
    "--clinical-csv",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/clinical_data.csv",
    show_default=True,
    help="Path to a CSV file of clinical data to pair with patients' data.",
)
@patient_data_params
@click.pass_context
def eval_population(
    ctx: click.Context,
    clinical_csv: Path,
    search_dirs: list[Path],
    graph_pattern: str,
    legacy_networkx_format: bool,
    ctpa_pattern: str | None = None,
) -> None:
    """Command to chain together loading data for a whole population with downstream tasks."""
    ctx.obj = {
        "graphs": {},
    }
    if ctpa_pattern:
        ctx.obj["ctpa_paths"] = {}

    log.info(f"Loading clinical data from {clinical_csv}...")
    data = load_and_clean_clinical_data(clinical_csv)
    ctx.obj["clinical_data"] = data  # Store loaded clinical data in the Click context object

    for patient_id in tqdm(data.index, desc="Loading patient files", unit="patient"):
        try:
            graph_file = find_file(patient_id, search_dirs=search_dirs, pattern=graph_pattern)
            ctpa_file = find_file(patient_id, search_dirs=search_dirs, pattern=ctpa_pattern) if ctpa_pattern else None
        except FileNotFoundError as e:
            if "No file found" in str(e):
                # Skip silently if no associated file is found
                continue
            if "Multiple matches" in str(e):
                # Log and skip if multiple files are found
                log.exception("", exc_info=True)
                continue
            # Re-raise unexpected errors
            raise

        graph: nx.DiGraph = json_to_networkx(graph_file, edges="links" if legacy_networkx_format else "edges")
        # Store the loaded data in the Click context object, to make them available to following commands in the chain
        ctx.obj["graphs"][patient_id] = graph
        if ctpa_file:
            ctx.obj["ctpa_paths"][patient_id] = ctpa_file

    if not ctx.obj["graphs"]:
        raise AssertionError(f"No graphs could be processed from directories: {search_dirs}.")


eval_population.add_command(qanadli)
eval_population.add_command(mastora)
eval_population.add_command(visualize)


@eval_population.command()
@click.option(
    "--col",
    "-c",
    "cols",
    type=click.Choice(list(PERSEVERE_ATTRS_LABELS.keys())),
    multiple=True,
    required=True,
    help="Attribute(s) to plot along the columns of the facet grid. If a graph score is specified, it must have been "
    "computed previously by calling their dedicated command (e.g. `qanadli`) earlier in the same command chain.",
)
@click.option(
    "--row",
    "-r",
    "rows",
    type=click.Choice(list(PERSEVERE_ATTRS_LABELS.keys())),
    multiple=True,
    required=True,
    help="Attribute(s) to plot along the rows of the facet grid. If a graph score is specified, it must have been "
    "computed previously by calling their dedicated command (e.g. `qanadli`) earlier in the same command chain.",
)
@click.option(
    "--categorical-plot",
    type=click.Choice(["violin", "histogram"]),
    default="violin",
    show_default=True,
    help="Type of plot to use for categorical attributes.",
)
@click.pass_obj
def plot(obj: dict, cols: list[str], rows: list[str], **facet_grid_kwargs) -> None:
    """Plot distribution of attribute(s) with respect to other attribute(s)."""
    # Recover the clinical data extracted by the main command
    if (data := obj.get("clinical_data")) is None:
        raise AssertionError(
            "Clinical data not found in the stored data. Unexpected error must have occurred in the main command."
        )

    # Recover measurable attributes if they were computed by a previous command in the chain
    # Otherwise, if they are not in the stored clinical data, raise an error to inform the user
    measurable_attributes = [attr for attr in cols + rows if attr in PERSEVERE_AUTO_MEASURES]
    for attr in measurable_attributes:
        if attr_data := obj.get(attr):
            # If the attribute was computed by a previous command in the chain, update the clinical data
            # Join the current score with the clinical data (and previous scores)
            data[attr] = pd.Series(attr_data["values"])

        if attr not in data.columns:
            raise ValueError(
                f"Attribute '{attr}' not found in the stored data; either call its dedicated command earlier in the "
                f"chain to compute it, or pre-compute it and add it to the clinical data CSV file."
            )

    script = os.path.basename(sys.argv[0])
    cli_cmd = f"{script} {' '.join(sys.argv[1:])}"

    log.info("Generating correlation plot...")
    facet_grid(data, cols, rows, cli_cmd, **facet_grid_kwargs)


if __name__ == "__main__":
    eval_population()
