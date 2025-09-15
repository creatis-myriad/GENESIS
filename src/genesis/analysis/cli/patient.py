from pathlib import Path

import click

from genesis.analysis.cli.commands import mastora, qanadli, visualize
from genesis.analysis.cli.parameters import patient_data_params
from genesis.analysis.cli.utils import get_logger
from genesis.data.utils import find_file, json_to_networkx

log = get_logger(__name__)


@click.group(chain=True)
@click.argument("patient_id", type=str)
@patient_data_params
@click.pass_context
def eval_patient(  # noqa: D417
    ctx: click.Context,
    patient_id: str,
    search_dirs: list[Path],
    graph_pattern: str,
    legacy_networkx_format: bool,
    ctpa_pattern: str | None = None,
) -> None:
    """Command to chain together loading a patient's data with downstream tasks (e.g. scoring, visualization).

    Args:
        patient_id: Patient ID (e.g. `0055`).
    """
    # Initialize the Click context object to store data across chained commands
    ctx.obj = {}

    graph_file = find_file(patient_id, search_dirs=search_dirs, pattern=graph_pattern)
    log.info(f"Loading graph from {graph_file}")
    graph = json_to_networkx(graph_file, edges="links" if legacy_networkx_format else "edges")
    # Store the loaded graph in the Click context object
    ctx.obj["graphs"] = {patient_id: graph}

    if ctpa_pattern:
        cpta_file = find_file(patient_id, search_dirs=search_dirs, pattern=ctpa_pattern)
        log.info(f"Saving path to CTPA {cpta_file}")
        # Store the CTPA path in the Click context object
        ctx.obj["ctpa_paths"] = {patient_id: cpta_file}


eval_patient.add_command(qanadli)
eval_patient.add_command(mastora)
eval_patient.add_command(visualize)


if __name__ == "__main__":
    eval_patient()
