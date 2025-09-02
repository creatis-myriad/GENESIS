from collections.abc import Callable
from pathlib import Path

import click
import rootutils


def graph_loading_params(func: Callable) -> Callable:
    """Decorator to add common CLI parameters to commands meant to load graph.

    This decorator injects the following parameters into the click command:
    - graphs_dir: Directory(ies) to search for graph files.
    - pattern: Glob pattern for locating the graph file.
    - legacy_networkx_format: If set, use legacy attribute names for NetworkX-internal graph data.

    Args:
        func: The click command function to decorate.

    Returns:
        The decorated function with additional click parameters.
    """
    func = click.option(
        "--graphs-dir",
        "-g",
        "graphs_dirs",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        multiple=True,
        default=[rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/graphs"],
        show_default=True,
        help="Directory(ies) to search for graph files.",
    )(func)
    func = click.option(
        "--pattern",
        "-p",
        type=str,
        default="*{id}*.json",
        show_default=False,
        help="Glob pattern to search for files within `graphs-dirs` (e.g. '*{id}_enriched_graph.json').",
    )(func)
    func = click.option(
        "--legacy-networkx-format",
        "-l",
        is_flag=True,
        default=False,
        help="Use legacy attribute names to parse NetworkX-internal graph data (i.e. 'links' instead of 'edges').",
    )(func)
    return func  # noqa: RET504
