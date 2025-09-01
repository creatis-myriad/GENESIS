from pathlib import Path

import click

from genesis.analysis.cli.commands import mastora, qanadli, visualize
from genesis.analysis.cli.parameters import graph_loading_params
from genesis.analysis.cli.utils import get_logger
from genesis.data.utils import find_graph_file, json_to_networkx

log = get_logger(__name__)


@click.group("eval-graph", chain=True)
@click.argument(
    "input-file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
@graph_loading_params
@click.pass_context
def eval_graph(  # noqa: D417
    ctx: click.Context, input_file: Path, graphs_dirs: list[Path], pattern: str, legacy_networkx_format: bool
) -> None:
    """Command to chain together loading a graph with downstream tasks (e.g. scoring, visualization).

    Args:
        input_file: Path to JSON graph or patient ID.
    """
    filepath = find_graph_file(input_file, search_dirs=graphs_dirs, pattern=pattern)
    log.info(f"Loading graph from {filepath}")
    graph = json_to_networkx(filepath, edges="links" if legacy_networkx_format else "edges")
    # Store the loaded graph in the Click context object, to make it available to following commands in the chain
    ctx.obj = {
        "graphs": {input_file: graph},
    }


eval_graph.add_command(qanadli)
eval_graph.add_command(mastora)
eval_graph.add_command(visualize)


if __name__ == "__main__":
    eval_graph()
