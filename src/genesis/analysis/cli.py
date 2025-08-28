import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import click
import rootutils

from genesis.analysis.analysis import correlate_and_plot, networkx_to_pyvis, pyvis_show
from genesis.analysis.scores.mastora import mastora as mastora_score
from genesis.analysis.scores.qanadli import qanadli as qanadli_score
from genesis.data.utils import find_graph_file, json_to_networkx

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_graph_loading_args(func: Callable) -> Callable:
    """Decorator to add common graph-related CLI arguments to commands meant to process a single graph.

    This decorator injects the following parameters into the click command:
    - input_file: Path or patient ID for the graph JSON.
    - graphs_dirs: Directories to search for graph files.
    - pattern: Glob pattern for locating the graph file.
    - legacy_networkx_format: If set, use legacy attribute names for NetworkX-internal graph data.
    - obstruction_attr: Edge attribute to use as obstruction values.

    Args:
        func: The click command function to decorate.

    Returns:
        The decorated function with additional click parameters.
    """
    func = click.argument(
        "input-file",
        type=click.Path(exists=False, dir_okay=False, path_type=Path),
    )(func)
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
    func = click.option(
        "--obstruction-attr",
        "-o",
        type=str,
        default="transversal_obstruction_max",
        show_default=True,
        help="The edge attribute to use for obstruction values.",
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


@click.command()
@add_graph_loading_args
@click.option(
    "--use-percentage",
    "-p",
    is_flag=True,
    default=False,
    help="If set, treat degrees as obstruction percentages [0, 1]. Otherwise, use degrees {1...5}.",
)
@click.option(
    "--mode",
    "-m",
    type=str,
    default="rmls",
    show_default=True,
    help="Artery levels to include: 'r' (root), 'm' (mediastinal), 'l' (lobar), 's' (segmental). "
    "Any combination (e.g., 'rmls').",
)
@click.option(
    "--debug", "-d", is_flag=True, default=False, help="If set, show a debug visualization of the Mastora calculation."
)
def mastora(
    use_percentage: bool,
    mode: str,
    debug: bool,
    **kwargs,
) -> None:
    """Compute Mastora score from a serialized graph file.

    Computes the Mastora score for pulmonary embolism risk assessment, evaluating the degree of vascular obstruction in
    mediastinal, lobar, and segmental arteries.
    """
    run_score(
        mastora_score,
        use_percentage=use_percentage,
        mode=mode,
        debug=debug,
        **kwargs,
    )


@click.command()
@add_graph_loading_args
@click.option(
    "--partial-obstruction-thresh",
    "-po",
    type=float,
    default=0.25,
    show_default=True,
    help="Transversal obstruction threshold to consider a segment partially obstructed.",
)
@click.option(
    "--total-obstruction-thresh",
    "-to",
    type=float,
    default=0.75,
    show_default=True,
    help="Transversal obstruction threshold to consider a segment totally obstructed.",
)
@click.option(
    "--debug", "-d", is_flag=True, default=False, help="If set, show a debug visualization of the Qanadli calculation."
)
def qanadli(
    partial_obstruction_thresh: float,
    total_obstruction_thresh: float,
    debug: bool,
    **kwargs,
) -> None:
    """Compute Qanadli score from a serialized graph file.

    Computes the Qanadli score for pulmonary embolism risk assessment, considering both embolus location and degree of
    obstruction, weighting each segment by its number of distal subsegments.
    """
    run_score(
        qanadli_score,
        partial_obstruction_thresh=partial_obstruction_thresh,
        total_obstruction_thresh=total_obstruction_thresh,
        debug=debug,
        **kwargs,
    )


@click.command()
@add_graph_loading_args
def visualize(
    input_file: Path, graphs_dirs: list[Path], pattern: str, legacy_networkx_format: bool, obstruction_attr: str
) -> None:
    """Visualize attribute values from a serialized graph file using PyVis.

    Creates an interactive network visualization of the arterial tree, coloring edges by the specified obstruction
    attribute.
    """
    filepath = find_graph_file(input_file, search_dirs=graphs_dirs, pattern=pattern)
    log.info(f"Loading graph from {filepath}")
    graph = json_to_networkx(filepath, edges="links" if legacy_networkx_format else "edges")
    log.info("Creating interactive visualization...")
    pyvis_show(networkx_to_pyvis(graph, attr=obstruction_attr), name=f"{filepath.stem}.html")
    log.info("Done. Open your browser to view it.")


@click.command()
@click.argument(
    "score",
    type=click.Choice(["mastora", "qanadli"], case_sensitive=False),
)
@click.argument(
    "target-attribute",
    type=click.Choice(["bnp", "troponin", "risk", "spesi"], case_sensitive=False),
)
@click.option(
    "--clinical-data",
    "-c",
    "clinical_data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/clinical_data.csv",
    show_default=True,
    help="Path to the clinical data CSV file.",
)
@click.option(
    "--graphs-dirs",
    "-g",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    default=[rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/raw"],
    show_default=True,
    help="Directory(ies) to search for graph files.",
)
@click.option(
    "--legacy-networkx-format",
    "-l",
    is_flag=True,
    default=False,
    help="If set, use legacy attribute name to parse NetworkX-internal graph data (i.e. 'links' instead of 'edges').",
)
@click.option(
    "--obstruction-attrs",
    "-o",
    type=str,
    multiple=True,
    default=["transversal_obstruction_max", "ancestors_obstruction_max", "ancestors_obstruction_cumulated"],
    show_default=True,
    help="Edge attribute(s) to use as obstruction values to compute global scores (i.e. Mastora, Qanadli).",
)
@click.option(
    "--show-visualization",
    "-v",
    is_flag=True,
    default=False,
    help="Open the correlation plot in a web browser.",
)
def correlate(  # noqa: D417
    score: Literal["mastora", "qanadli"],
    target_attribute: Literal["bnp", "troponin", "risk", "spesi"],
    clinical_data_path: Path,
    graphs_dirs: list[Path],
    legacy_networkx_format: bool,
    obstruction_attrs: list[str],
    show_visualization: bool,
) -> None:
    """Correlate global vascular tree obstruction scores with clinical attributes and visualize the results.

    Args:
        score: The global obstruction score to compute and correlate with `target_attribute`.
        target_attribute: The clinical attribute to correlate with the computed `score`.
    """
    script = os.path.basename(sys.argv[0])
    cli_cmd = f"{script} {' '.join(sys.argv[1:])}"
    correlate_and_plot(
        score=score,
        target_attribute=target_attribute,
        clinical_data_path=clinical_data_path,
        graphs_dirs=list(graphs_dirs),
        legacy_networkx_format=legacy_networkx_format,
        obstruction_attrs=obstruction_attrs,
        cli_command=cli_cmd,
        show_visualization=show_visualization,
    )
