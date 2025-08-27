import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import networkx as nx

from genesis.data.utils import networkx_aggregate_attrs, networkx_has_edge_attributes
from genesis.utils.builtin import is_identical_function


def derive_missing_obstruction_attrs(graph_arg: int | str, attrs_args: list[int | str]) -> Callable:
    """Decorator to derive graph obstruction attributes on-the-fly from other attributes.

    The decorator ensures that the requested obstruction attributes are available for the decorated function.
    If the obstruction attributes are already present in the graph, they are not recomputed.

    Args:
        graph_arg: Arg pos or kwarg name in the decorated function that contains the NetworkX graph.
        attrs_args: Arg pos or kwarg names in the decorated function that contain obstruction attribute(s) to make sure
            are present in the graph.

    Notes:
        - The decorated function must support an `in_place` kwarg to specify whether to modify the input graph in place
          or return a modified copy.

    Returns:
        Decorated function where the requested obstruction attributes are ensured to be present in the graph.
    """

    def _derive_missing_obstruction_attrs_decorator(func: Callable) -> Callable:
        @wraps(func)
        def _derive_and_call(*args, **kwargs) -> Any:
            # Import locally to avoid circular imports
            from genesis.analysis.scores.mastora import mastora  # noqa: PLC0415
            from genesis.analysis.scores.qanadli import qanadli  # noqa: PLC0415
            from genesis.analysis.scores.vascular_tree import (  # noqa: PLC0415
                ancestors_obstruction_cumulated,
                ancestors_obstruction_max,
            )

            def _get_arg_val(pos_or_name: int | str) -> Any:
                if isinstance(pos_or_name, int):
                    if pos_or_name < len(args):
                        return args[pos_or_name]
                    raise IndexError(f"Argument position {pos_or_name} out of range for function '{func.__name__}'")
                if isinstance(pos_or_name, str):
                    if kwarg := kwargs.get(pos_or_name):  # First check in supplied kwargs
                        return kwarg
                    # If not in kwargs, check in function signature parameters not overridden by supplied args
                    if parameter := inspect.signature(func).parameters.get(pos_or_name):
                        return parameter.default
                    raise KeyError(f"Keyword argument '{pos_or_name}' not found for function '{func.__name__}'")
                raise TypeError(
                    f"Argument specifier {pos_or_name} in `derive_missing_obstruction_attrs` decorator must be an "
                    f"int (positional arg) or str (keyword arg)."
                )

            graph: nx.DiGraph = _get_arg_val(graph_arg)
            attrs: list[str] = []
            for attrs_arg in attrs_args:  # For each argument that may contain one or more attributes
                arg_val = _get_arg_val(attrs_arg)
                if isinstance(arg_val, str):  # If single attribute, append it
                    attrs.append(arg_val)
                else:  # If list of attributes, extend the list
                    attrs.extend(arg_val)

            # Forward the `in_place` kwarg if present, to make sure attributes derived recursively follow the same
            # graph update logic
            in_place = kwargs.get("in_place", False)
            # For Mastora and Qanadli score functions, if debugging, force in place graph update to keep derived
            # attributes for visualization
            if is_identical_function(func, mastora) or is_identical_function(func, qanadli):
                in_place = in_place or kwargs.get("debug")

            for attr in attrs:
                # Only compute attribute if it is not available
                if not networkx_has_edge_attributes(graph, attrs=[attr]):
                    match attr:
                        case "ancestors_obstruction_max":
                            graph = ancestors_obstruction_max(graph, in_place=in_place)
                        case "ancestors_obstruction_cumulated":
                            graph = ancestors_obstruction_cumulated(graph, in_place=in_place)
                        case _:  # If not a special case, aggregate existing attributes based on name and suffix
                            # Determine base attribute and aggregation function from the attribute's name
                            base_attr, agg = attr.rsplit("_", 1)
                            graph = networkx_aggregate_attrs(
                                graph, {base_attr: agg}, element="edges", in_place=in_place
                            )

                if isinstance(graph_arg, int):
                    args = list(args)
                    args[graph_arg] = graph
                else:
                    kwargs[graph_arg] = graph

            return func(*args, **kwargs)

        return _derive_and_call

    return _derive_missing_obstruction_attrs_decorator
