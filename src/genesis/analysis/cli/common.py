import logging
from collections.abc import Callable
from pathlib import Path

import click
import rootutils

from genesis.analysis.analysis import networkx_to_pyvis, pyvis_show
from genesis.data.utils import find_graph_file, json_to_networkx


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name and a standardized configuration."""
    log = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO)
    return log


log = get_logger(__name__)


def graph_loading_params(func: Callable) -> Callable:
    """Decorator to add common CLI parameters to commands meant to load graph.

    This decorator injects the following parameters into the click command:
    - graphs_dirs: Directories to search for graph files.
    - pattern: Glob pattern for locating the graph file.
    - legacy_networkx_format: If set, use legacy attribute names for NetworkX-internal graph data.

    Args:
        func: The click command function to decorate.

    Returns:
        The decorated function with additional click parameters.
    """
    func = click.option(
        "--graphs-dirs",
        "-g",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        multiple=True,
        default=[rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/raw"],
        show_default=True,
        help="Directory(ies) to search for graph files.",
    )(func)
    func = click.option(
        "--pattern",
        "-P",
        type=str,
        default="*{id}*.json",
        show_default=False,
        help="(internal) glob pattern, e.g. '*{id}_enriched.json'.",
    )(func)
    func = click.option(
        "--legacy-networkx-format",
        "-l",
        is_flag=True,
        default=False,
        help="If set, use legacy attribute name to parse NetworkX-internal graph data "
        "(i.e. 'links' instead of 'edges').",
    )(func)
    return func  # noqa: RET504


def score_params(func: Callable) -> Callable:
    """Decorator to add common CLI parameters to commands meant to compute global graph scores.

    This decorator injects the following parameters into the click command:
    - obstruction_attr: Edge attribute to use as obstruction values.
    - debug: If set, show a debug visualization of the score calculation.

    Args:
        func: The click command function to decorate.

    Returns:
        The decorated function with additional click parameters.
    """
    func = click.option(
        "--obstruction-attr",
        "-o",
        type=str,
        default="transversal_obstruction_max",
        show_default=True,
        help="The edge attribute to use for obstruction values.",
    )(func)
    func = click.option(
        "--debug",
        "-d",
        is_flag=True,
        default=False,
        help="If set, show a debug visualization of the score calculation.",
    )(func)
    return func  # noqa: RET504


def run_score(
    score_fn: Callable,
    input_file: Path,
    graphs_dirs: list[Path],
    pattern: str,
    legacy_networkx_format: bool,
    obstruction_attr: str,
    **score_kwargs,
) -> None:
    """Common runner for score-based CLI commands.

    Loads a graph file, computes the score, and optionally displays a visualization for debugging.

    Args:
        score_fn: The global obstruction score function.
        input_file: Path to JSON graph or patient ID.
        graphs_dirs: Directories to search for graph files.
        pattern: Glob pattern for locating the graph file.
        legacy_networkx_format: If set, use legacy attribute names for NetworkX-internal graph data.
        obstruction_attr: Edge attribute for obstruction values.
        **score_kwargs: Additional parameters to pass along to `score_fn`.
    """
    filepath = find_graph_file(input_file, search_dirs=graphs_dirs, pattern=pattern)
    log.info(f"Loading graph from {filepath}")
    graph = json_to_networkx(filepath, edges="links" if legacy_networkx_format else "edges")
    log.info(f"Computing {score_fn.__name__} score...")
    if score_kwargs.pop("debug", False):
        score, dbg_info = score_fn(graph, obstruction_attr=obstruction_attr, debug=True, **score_kwargs)
        log.info(f"Score: {score}")
        log.info("Creating interactive visualization with debug info...")
        pyvis_net = networkx_to_pyvis(graph, attr=obstruction_attr, debug_info=dbg_info)
        pyvis_show(pyvis_net, name=f"{filepath.stem}_{score_fn.__name__}.html")
        log.info("Done. Open your browser to view it.")
    else:
        score = score_fn(graph, obstruction_attr=obstruction_attr, **score_kwargs)
        log.info(f"Score: {score}")
