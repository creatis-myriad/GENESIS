from pathlib import Path

import click

from genesis.analysis.cli.common import graph_loading_params, run_score, score_params
from genesis.analysis.scores.mastora import mastora as mastora_score


@click.command()
@graph_loading_params
@score_params
@click.argument(
    "input-file",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
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
def mastora(  # noqa: D417
    input_file: Path,
    use_percentage: bool,
    mode: str,
    **kwargs,
) -> None:
    """Compute Mastora score from a serialized graph file.

    Computes the Mastora score for pulmonary embolism risk assessment, evaluating the degree of vascular obstruction in
    mediastinal, lobar, and segmental arteries.

    Args:
        input_file: Path to JSON graph or patient ID.
    """
    run_score(
        mastora_score,
        input_file,
        use_percentage=use_percentage,
        mode=mode,
        **kwargs,
    )


if __name__ == "__main__":
    mastora()
