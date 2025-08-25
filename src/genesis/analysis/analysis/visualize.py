import tempfile

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


def pyvis_show(net: Network, notebook: bool = False) -> None:
    """Render a PyVis Network visualization.

    Utility function to wrap the creation of a temporary HTML file to save the generated HTML visualization.

    Args:
        net: The PyVis Network instance to render.
        notebook: Whether to render inline in a Jupyter notebook (True) or as a standalone HTML file (False).
    """
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as fp:
        net.show(fp.name, notebook=notebook)


def networkx_to_pyvis(
    graph: nx.DiGraph,
    attr: str = "transversal_obstruction_max",
    level_attr: str = "level",
    use_hierarchical: bool = True,
    height: str = "1400px",
    width: str = "100%",
    bg_color: str = "#000000",
    font_color: str = "#ffffff",
    min_edge_width: float = 0.4,
    max_edge_width: float = 30.0,
    debug_edges: list[tuple] | None = None,
    debug_labels: list[str] | None = None,
) -> Network:
    """Converts a directed NetworkX graph with attribute values to a PyVis Network, which can be visualized as HTML.

    Graph edges are colored based on their attribute values using a yellow-to-red
    colormap and sized (inversely) based on their `level` attribute.
    Node layout can be hierarchical or force-directed.
    Optionally, display debug labels near specified edges for algorithm tracing.

    Args:
        graph: A NetworkX directed graph to visualize.
        attr: The edge attribute to display.
        level_attr: The edge attribute to display as level (for width).
        use_hierarchical: Whether to use hierarchical layout (True) or force-directed layout (False).
        height: Height of the visualization container.
        width: Width of the visualization container.
        bg_color: Background color in hex format.
        font_color: Font color in hex format.
        min_edge_width: Minimum edge width for highest levels.
        max_edge_width: Maximum edge width for lowest levels.
        debug_edges: List of edge tuples (u, v) to annotate with labels.
        debug_labels: List of debug label strings corresponding to `debug_edges`.

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
    )
    if use_hierarchical:
        net.set_options(HIERARCHICAL_LAYOUT_OPTIONS)

    # Add nodes from the NetworkX graph to the PyVis Network
    _add_nodes(net, graph)

    # build debug map if annotations provided
    debug_map: dict[tuple, str] = {}
    if debug_edges is not None or debug_labels is not None:
        if not debug_edges or not debug_labels or len(debug_edges) != len(debug_labels):
            raise ValueError("`debug_edges` and `debug_labels` must both be provided and of equal length.")
        debug_map = dict(zip(debug_edges, debug_labels, strict=False))

    # Prepare normalizers and colormap for edge styling
    attr_norm, level_norm, attr_cmap = _init_color_and_level_normalizers(graph, attr, level_attr)
    # Add styled edges from the NetworkX graph to the PyVis Network
    _add_edges(
        net,
        graph,
        attr,
        level_attr,
        attr_norm,
        level_norm,
        attr_cmap,
        min_edge_width,
        max_edge_width,
        debug_map,
    )

    return net


def _init_color_and_level_normalizers(
    graph: nx.DiGraph,
    attr: str,
    level_attr: str,
) -> tuple[Normalize, Normalize, LinearSegmentedColormap]:
    """Compute normalizers and colormap for edge coloring and sizing.

    Args:
        graph: A NetworkX directed graph.
        attr: The edge attribute to display.
        level_attr: The edge attribute to display as level.

    Returns:
        A tuple (attr_norm, level_norm, attr_cmap) where:
            attr_norm: Normalize instance for attribute values.
            level_norm: Normalize instance for level values.
            attr_cmap: Colormap for attribute-to-color mapping.
    """
    attr_vals = [data.get(attr, 0.0) for _, _, data in graph.edges(data=True)]
    attr_norm = Normalize(vmin=min(attr_vals, default=0.0), vmax=max(attr_vals, default=1.0) or 1.0)
    attr_cmap = LinearSegmentedColormap.from_list("bpr", ["#aaaaff", "#ff00ff", "#ff0000"])

    level_vals = [data.get(level_attr, 0.0) for _, _, data in graph.edges(data=True)]
    level_norm = Normalize(vmin=min(level_vals, default=0.0), vmax=max(level_vals, default=1.0) or 1.0)
    return attr_norm, level_norm, attr_cmap


def _add_nodes(net: Network, graph: nx.DiGraph) -> None:
    """Add nodes from a NetworkX graph to a PyVis Network.

    Args:
        net: The PyVis Network to which nodes will be added.
        graph: A NetworkX directed graph.
    """
    for n in graph.nodes():
        net.add_node(n, label=str(n))


def _add_edges(
    net: Network,
    graph: nx.DiGraph,
    attr: str,
    level_attr: str,
    attr_norm: Normalize,
    level_norm: Normalize,
    attr_cmap: LinearSegmentedColormap,
    min_edge_width: float,
    max_edge_width: float,
    debug_map: dict[tuple, str] | None = None,
) -> None:
    """Add styled edges from a NetworkX graph to a PyVis Network.

    Edge colors are based on obstruction values and widths inversely on level values.
    Optional debug labels can be displayed on specified edges.

    Args:
        net: The PyVis Network to which edges will be added.
        graph: A NetworkX directed graph.
        attr: The edge attribute name containing obstruction values.
        level_attr: The edge attribute name containing level values.
        attr_norm: Normalizer for obstruction values.
        level_norm: Normalizer for level values.
        attr_cmap: Colormap for mapping normalized obstruction to RGB.
        min_edge_width: Minimum edge width for highest levels.
        max_edge_width: Maximum edge width for lowest levels.
        debug_map: Optional mapping from edge (u, v) tuples to debug label strings.
    """
    debug_map = debug_map or {}
    for u, v, data in graph.edges(data=True):
        obs = data.get(attr, 0.0)
        r, g, b, _ = attr_cmap(attr_norm(obs))
        color = f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"

        lvl = data.get(level_attr, 0.0)
        inv = 1.0 - level_norm(lvl)
        width = min_edge_width + (max_edge_width - min_edge_width) * inv

        edge_kwargs = {
            "color": color,
            "width": width,
            "title": f"{attr}: {obs:.2f} | {level_attr}: {lvl}",
            "arrows": "to",
        }
        if (u, v) in debug_map:
            edge_kwargs["label"] = debug_map[(u, v)]
            if (u, v) in debug_map:
                edge_kwargs["label"] = debug_map[(u, v)]
                edge_kwargs["font"] = {"size": 33, "color": "#ffffff", "strokeWidth": 0, "align": "top", "vadjust": -50}
        net.add_edge(u, v, **edge_kwargs)
