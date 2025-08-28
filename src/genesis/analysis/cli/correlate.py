import os
import sys
from pathlib import Path
from typing import Literal

import click
import rootutils

from genesis.analysis.analysis import correlate_and_plot
from genesis.analysis.cli.common import graph_loading_params


@click.command()
@graph_loading_params
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
    obstruction_attrs: list[str],
    show_visualization: bool,
    **kwargs,
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
        obstruction_attrs=obstruction_attrs,
        cli_command=cli_cmd,
        show_visualization=show_visualization,
        **kwargs,
    )


if __name__ == "__main__":
    correlate()
