from pathlib import Path

import click

from genesis.analysis.cli.common import graph_loading_params, run_score, score_params
from genesis.analysis.scores.qanadli import qanadli as qanadli_score


@click.command()
@graph_loading_params
@score_params
@click.argument(
    "input-file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
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
def qanadli(  # noqa: D417
    input_file: Path,
    partial_obstruction_thresh: float,
    total_obstruction_thresh: float,
    **kwargs,
) -> None:
    """Compute Qanadli score from a serialized graph file.

    Computes the Qanadli score for pulmonary embolism risk assessment, considering both embolus location and degree of
    obstruction, weighting each segment by its number of distal subsegments.

    Args:
        input_file: Path to JSON graph or patient ID.
    """
    run_score(
        qanadli_score,
        input_file,
        partial_obstruction_thresh=partial_obstruction_thresh,
        total_obstruction_thresh=total_obstruction_thresh,
        **kwargs,
    )


if __name__ == "__main__":
    qanadli()
