from collections.abc import Callable
from pathlib import Path

import click
import rootutils


def patient_data_params(func: Callable) -> Callable:
    """Decorator to add common CLI parameters to commands meant to load patient data.

    This decorator injects the following parameters into the click command:
    - search_dir: Directory(ies) to search for input files.
    - ctpa_pattern: Glob pattern for locating the CTPA image file.
    - graph_pattern: Glob pattern for locating the graph file.
    - legacy_networkx_format: If set, use legacy attribute names for NetworkX-internal graph data.

    Args:
        func: The click command function to decorate.

    Returns:
        The decorated function with additional click parameters.
    """
    func = click.option(
        "--search-dir",
        "-d",
        "search_dirs",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        multiple=True,
        default=[rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/graphs"],
        show_default=True,
        help="Directory(ies) to search for input files.",
    )(func)
    func = click.option(
        "--graph-pattern",
        "-g",
        type=str,
        default="*{id}*.json",
        show_default=False,
        help="Glob pattern to search for vascular graph JSON files within `search-dir`.",
    )(func)
    func = click.option(
        "--legacy-networkx-format",
        "-l",
        is_flag=True,
        default=False,
        help="Use legacy attribute names to parse NetworkX-internal graph data (i.e. 'links' instead of 'edges').",
    )(func)
    func = click.option(
        "--ctpa-pattern",
        "-i",
        type=str,
        show_default=False,
        help="Glob pattern to search for CPTA image files within `search-dir`.",
    )(func)
    return func  # noqa: RET504
