from collections.abc import Callable
from pathlib import Path
from typing import Literal

import click
from click import pass_obj
from tqdm.auto import tqdm

from genesis.analysis.cli.utils import get_logger
from genesis.analysis.plot.graph import networkx_to_pyvis
from genesis.analysis.scores.mastora import mastora as mastora_score
from genesis.analysis.scores.qanadli import qanadli as qanadli_score
from genesis.data.utils import networkx_has_attributes

log = get_logger(__name__)

# Options common to scoring commands
option_obstruction_attr = click.option(
    "--obstruction-attr",
    "-o",
    type=str,
    default="transversal_obstruction_max",
    show_default=True,
    help="Edge attribute to use as obstruction values.",
)


@click.command()
@option_obstruction_attr
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
@click.pass_obj
def qanadli(obj: dict, *args, **kwargs) -> None:
    """Compute Qanadli score on the graph(s).

    Computes the Qanadli score for pulmonary embolism risk assessment, considering both embolus location and degree of
    obstruction, weighting each segment by its number of distal subsegments.
    """
    _run_score(obj, qanadli_score, *args, **kwargs)


@click.command()
@click.argument(
    "mode",
    type=click.Choice(["central", "peripheral", "global"]),
)
@option_obstruction_attr
@click.pass_obj
def mastora(obj: dict, mode: Literal["central", "peripheral", "global"], **kwargs) -> None:  # noqa: D417
    """Compute Mastora score on the graph(s).

    Computes the Mastora score for pulmonary embolism risk assessment, evaluating the precise degree of vascular
    obstruction in central and/or peripheral arteries.

    Args:
        mode: Variant of the Mastora score to compute.
            - 'central': Considers obstructions in the mediastinal and lobar arteries
            - 'peripheral': Considers obstructions in the segmental arteries
            - 'global': Considers obstructions in all arteries (i.e. both central and peripheral)
    """
    _run_score(obj, mastora_score, mode, score_name=f"mastora_{mode}", **kwargs)


def _run_score(
    obj: dict,
    score_fn: Callable,
    *score_args,
    score_name: str | None = None,
    obstruction_attr: str = "transversal_obstruction_max",
    **score_kwargs,
) -> None:
    """Compute a global score on a graph (previously loaded in the click context), with optional debug visualization.

    Args:
        obj: State dict to store objects and communicate between commands, part of the Click context.
        score_fn: The global obstruction score function.
        *score_args: Positional arguments to pass along to `score_fn`.
        score_name: Name to use for the score in logging and storing in `obj`. If None, uses `score_fn.__name__`.
        obstruction_attr: Edge attribute to use as obstruction values.
        **score_kwargs: Additional parameters to pass along to `score_fn`.
    """
    score_name = score_name or score_fn.__name__

    graphs = obj["graphs"]
    scores = {}
    debug_info = {}
    for graph_file, graph in tqdm(graphs.items(), desc=f"Compute {score_name} score on input graphs", unit="graph"):
        score, graph_debug_info = score_fn(
            graph, *score_args, obstruction_attr=obstruction_attr, debug=True, **score_kwargs
        )
        scores[graph_file] = score
        debug_info[graph_file] = graph_debug_info

    obj[score_name] = {
        "scores": scores,
        "debug_info": debug_info,
        "obstruction_attr": obstruction_attr,
    }
    # If only one graph was processed, show its score and optionally a debug visualization
    if len(scores) == 1:
        graph_file = next(iter(scores))
        log.info(f"{score_name} score for graph {graph_file}: {scores[graph_file]}")


@click.command()
@click.option(
    "--color-attr",
    "-c",
    type=str,
    help="Edge attribute to color edges in the graph visualization. Ignored in favor of obstruction attribute used "
    "by score if `--debug-score` is also specified.",
)
@click.option(
    "--debug-score",
    "-d",
    type=click.Choice(["qanadli", "mastora_central", "mastora_peripheral", "mastora_global"]),
    help="Score for which to display intermediate values in the visualization, to help debugging.",
)
@click.option(
    "--output-dir",
    "-O",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Directory to save the generated HTML files under.",
)
@pass_obj
def visualize(
    obj: dict,
    color_attr: str | None = None,
    debug_score: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Visualize attribute values in the graph(s) using an interactive PyVis-generated HTML.

    Creates an interactive network visualization of the arterial tree, coloring edges by the specified attribute.
    """
    if score_data := obj.get(debug_score):
        if color_attr:
            log.warning(
                f"Both `--color-attr` and `--debug-score` are specified. '{color_attr}' value "
                f"(`--color-attr`) ignored in favor of '{score_data['obstruction_attr']}' attribute used by "
                f"{debug_score} (`--debug-score`)."
            )
        color_attr = score_data["obstruction_attr"]

    output_dir = output_dir or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    for graph_file, graph in tqdm(
        obj["graphs"].items(), desc="Generating interactive visualizations for input graphs", unit="graph"
    ):
        if color_attr and not networkx_has_attributes(graph, element="edges", attrs=[color_attr]):
            raise ValueError(
                f"Graph {graph_file} does not have edge attribute '{color_attr}', required for visualization."
            )

        visu_filename = f"{graph_file.stem}.html"
        debug_info = None
        if debug_score:
            debug_info = score_data["debug_info"][graph_file]
            visu_filename = f"{graph_file.stem}_{color_attr}_{debug_score}.html"

        net = networkx_to_pyvis(graph, color_attr=color_attr, debug_info=debug_info)
        net.show(str(output_dir / visu_filename), notebook=False)
