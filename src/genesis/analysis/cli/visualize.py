from pathlib import Path

import click

from genesis.analysis.analysis import networkx_to_pyvis, pyvis_show
from genesis.analysis.cli.common import get_logger, graph_loading_params
from genesis.data.utils import find_graph_file, json_to_networkx

log = get_logger(__name__)


@click.command()
@graph_loading_params
@click.argument(
    "input-file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
@click.option(
    "--obstruction-attr",
    "-o",
    type=str,
    default="transversal_obstruction_max",
    show_default=True,
    help="The edge attribute to use for obstruction values.",
)
def visualize(  # noqa: D417
    input_file: Path, graphs_dirs: list[Path], pattern: str, legacy_networkx_format: bool, obstruction_attr: str
) -> None:
    """Visualize attribute values from a serialized graph file using PyVis.

    Creates an interactive network visualization of the arterial tree, coloring edges by the specified obstruction
    attribute.

    Args:
        input_file: Path to JSON graph or patient ID.
    """
    filepath = find_graph_file(input_file, search_dirs=graphs_dirs, pattern=pattern)
    log.info(f"Loading graph from {filepath}")
    graph = json_to_networkx(filepath, edges="links" if legacy_networkx_format else "edges")
    log.info("Creating interactive visualization...")
    pyvis_show(networkx_to_pyvis(graph, attr=obstruction_attr), name=f"{filepath.stem}.html")
    log.info("Done. Open your browser to view it.")


if __name__ == "__main__":
    visualize()
