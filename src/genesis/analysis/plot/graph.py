import networkx as nx
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pyvis.network import Network

HIERARCHICAL_LAYOUT_OPTIONS = """
{
  "layout": { "hierarchical": {
    "enabled": true,
    "direction": "UD",
    "levelSeparation": 200,
    "nodeSpacing": 150,
    "treeSpacing": 300
  }},
  "physics": { "enabled": false }
}
"""


def networkx_to_pyvis(
    graph: nx.DiGraph,
    use_hierarchical: bool = True,
    height: str = "1400px",
    width: str = "100%",
    bg_color: str = "#000000",
    font_color: str = "#ffffff",
    width_attr: str = "level",
    min_edge_width: float = 0.4,
    max_edge_width: float = 30.0,
    color_attr: str | None = None,
    debug_info: dict[tuple[int, int], str] | None = None,
    **network_init_kwargs,
) -> Network:
    """Converts a directed NetworkX graph with attribute values to a PyVis Network, which can be visualized as HTML.

    Args:
        graph: A NetworkX directed graph to visualize.
        use_hierarchical: Whether to use hierarchical layout (True) or force-directed layout (False).
        height: Height of the visualization container.
        width: Width of the visualization container.
        bg_color: Background color in hex format.
        font_color: Font color in hex format.
        width_attr: The edge attribute used to determine edge width, from thick (low values) to thin (high values).
        min_edge_width: Minimum edge width.
        max_edge_width: Maximum edge width.
        color_attr: The edge attribute used to determine edge color, from blue (low values) to red (high values).
        debug_info: Mapping between edge tuples (u, v) and their debug annotations to display on the edges.
        **network_init_kwargs: Additional keyword arguments to forward to the PyVis Network init.

    Returns:
        A PyVis Network instance representing the visualization.
    """
    # Initialize PyVis Network
    net = Network(
        height=height,
        width=width,
        bgcolor=bg_color,
        font_color=font_color,
        directed=True,
        notebook=False,
        cdn_resources="remote",
        **network_init_kwargs,
    )
    if use_hierarchical:
        net.set_options(HIERARCHICAL_LAYOUT_OPTIONS)

    # Add nodes from the NetworkX graph to the PyVis Network
    _add_nodes(net, graph)

    # Add styled edges from the NetworkX graph to the PyVis Network
    _add_edges(
        net,
        graph,
        width_attr,
        min_edge_width,
        max_edge_width,
        color_attr=color_attr,
        debug_info=debug_info,
    )

    return net


def _add_nodes(net: Network, graph: nx.DiGraph) -> None:
    """Add nodes from a NetworkX graph to a PyVis Network.

    Args:
        net: The PyVis Network to which nodes will be added.
        graph: A NetworkX directed graph whose nodes to add to the PyVis Network.
    """
    for n in graph.nodes():
        net.add_node(n, label=str(n))


def _add_edges(
    net: Network,
    graph: nx.DiGraph,
    width_attr: str,
    min_edge_width: float,
    max_edge_width: float,
    color_attr: str | None = None,
    debug_info: dict[tuple, str] | None = None,
) -> None:
    """Add styled edges from a NetworkX graph to a PyVis Network.

    Args:
        net: The PyVis Network to which edges will be added.
        graph: The NetworkX directed graph whose edge to add to the PyVis Network.
        width_attr: The edge attribute used to determine edge width, from thick (low values) to thin (high values).
        min_edge_width: Minimum edge width.
        max_edge_width: Maximum edge width.
        color_attr: The edge attribute used to determine edge color, from blue (low values) to red (high values).
        debug_info: Mapping between edge tuples (u, v) and their debug annotations to display on the edges.
    """
    # Prepare normalizers and colormap for edge styling
    width_norm = _init_normalizer(graph, width_attr)
    cmap = LinearSegmentedColormap.from_list("bpr", ["#aaaaff", "#ff00ff", "#ff0000"])
    color_norm = _init_normalizer(graph, color_attr) if color_attr else None

    debug_info = debug_info or {}
    for u, v, data in graph.edges(data=True):
        # Determine edge width
        width_attr_vals = data[width_attr]
        inv = 1.0 - width_norm(width_attr_vals)
        width = min_edge_width + (max_edge_width - min_edge_width) * inv

        # Add always-on customization to edge kwargs
        title = f"{width_attr}: {width_attr_vals}"
        edge_kwargs = {
            "width": width,
            "arrows": "to",
        }

        # Determine edge color, if applicable
        if color_attr:
            color_attr_vals = data[color_attr]
            r, g, b, _ = cmap(color_norm(color_attr_vals))
            color = f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"

            edge_kwargs["color"] = color

            title += f" | {color_attr}: {color_attr_vals:.2f}"

        # Add debug info to edge kwargs, if applicable
        if (edge_debug_info := debug_info.get((u, v))) is not None:
            edge_kwargs["label"] = edge_debug_info
            edge_kwargs["font"] = {"size": 33, "color": "#ffffff", "strokeWidth": 0, "align": "top", "vadjust": -50}

        net.add_edge(u, v, title=title, **edge_kwargs)


def _init_normalizer(graph: nx.DiGraph, attr: str) -> Normalize:
    """Compute a normalizer for edge attribute values.

    Args:
        graph: A NetworkX directed graph.
        attr: The edge attribute to normalize.

    Returns:
        A Normalize instance for attribute values.
    """
    attr_vals = [data[attr] for _, _, data in graph.edges(data=True)]
    return Normalize(vmin=min(attr_vals), vmax=max(attr_vals) or 1.0)
