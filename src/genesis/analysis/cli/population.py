import os
import sys
from pathlib import Path

import click
import networkx as nx
import pandas as pd
import rootutils
from tqdm.auto import tqdm

from genesis.analysis.analysis.correlate import plot_correlation
from genesis.analysis.cli.commands import mastora, qanadli, visualize
from genesis.analysis.cli.parameters import graph_loading_params
from genesis.analysis.cli.utils import get_logger
from genesis.data.utils.io import find_graph_file, json_to_networkx, load_and_clean_clinical_data

log = get_logger(__name__)


@click.group("eval-graph", chain=True)
@click.option(
    "--clinical-csv",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/clinical_data.csv",
    show_default=True,
    help="Path to a CSV file of clinical data to pair with patients' graphs.",
)
@graph_loading_params
@click.pass_context
def eval_population(
    ctx: click.Context, clinical_csv: Path, graphs_dirs: list[Path], pattern: str, legacy_networkx_format: bool
) -> None:
    """Command to chain together loading clinical data and graphs for a whole population, for downstream tasks."""
    ctx.obj = {
        "graphs": {},
    }

    log.info(f"Loading clinical data from {clinical_csv}...")
    data = load_and_clean_clinical_data(clinical_csv)
    ctx.obj["clinical_data"] = data  # Store loaded clinical data in the Click context object

    for patient_id in tqdm(data.index, desc="Loading vascular graphs", unit="graph"):
        try:
            graph_file = find_graph_file(patient_id, search_dirs=graphs_dirs, pattern=pattern)
        except FileNotFoundError:
            continue  # Skip patient if no associated vascular tree graph is found
        except RuntimeError:
            # Log and skip if multiple files are found
            log.exception("", exc_info=True)
            continue

        graph: nx.DiGraph = json_to_networkx(graph_file, edges="links" if legacy_networkx_format else "edges")
        # Store the loaded graphs in the Click context object, to make them available to following commands in the chain
        ctx.obj["graphs"][patient_id] = graph

    if not ctx.obj["graphs"]:
        raise AssertionError(f"No graphs could be processed from directories: {graphs_dirs}.")


eval_population.add_command(qanadli)
eval_population.add_command(mastora)
eval_population.add_command(visualize)


@eval_population.command()
@click.option(
    "--score",
    "-s",
    "scores",
    type=click.Choice(["qanadli", "mastora_central", "mastora_peripheral", "mastora_global"]),
    multiple=True,
    required=True,
    help="The global graph score(s) to correlate with `clinical_attr`. They must have been computed previously by "
    "calling their dedicated command (e.g. `qanadli`) earlier in the same command chain.",
)
@click.option(
    "--clinical-attr",
    "-c",
    "clinical_attrs",
    type=click.Choice(["bnp", "troponin", "risk", "spesi"], case_sensitive=False),
    multiple=True,
    required=True,
    help="The clinical attribute(s) to correlate with `score`.",
)
@click.pass_obj
def correlate(obj: dict, scores: list[str], clinical_attrs: list[str]) -> None:
    """Correlate graph scores with clinical data."""
    # Recover the clinical data extracted by the main command
    if (data := obj.get("clinical_data")) is None:
        raise AssertionError(
            "Clinical data not found in the stored data. Unexpected error must have occurred in the main command."
        )

    # Recover the requested scores, checking they were computed by a previous command in the chain
    for score in scores:
        if not (score_dict := obj.get(score, {}).get("scores")):
            raise ValueError(
                f"Score '{score}' not found in the stored data; please call its dedicated command earlier in the chain "
                f"to compute it."
            )
        # Join the current score with the clinical data (and previous scores)
        data = data.join(pd.Series(score_dict, name=score))

    script = os.path.basename(sys.argv[0])
    cli_cmd = f"{script} {' '.join(sys.argv[1:])}"

    log.info("Generating correlation plot...")
    plot_correlation(data, scores, clinical_attrs, cli_cmd)


if __name__ == "__main__":
    eval_population()
