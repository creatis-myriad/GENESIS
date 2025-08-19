import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import networkx as nx
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pyvis.network import Network


def visualize_attribute_graph_pyvis(
    graph: nx.DiGraph,
    obstruction_attr: str = "max_transversal_obstruction_cumulated",
    level_attr: str = "level",
    use_hierarchical: bool = True,
    height: str = "1400px",
    width: str = "100%",
    bgcolor: str = "#000000",
    font_color: str = "#ffffff",
    min_edge_width: float = 0.4,
    max_edge_width: float = 30.0,
    debug_edges: list[tuple] | None = None,
    debug_labels: list[str] | None = None,
) -> None:
    """Visualizes a directed graph with attribute values using PyVis.

    Creates a temporary HTTP server to render the graph visualization in a web browser.
    Graph edges are colored based on their attribute values using a yellow-to-red
    colormap and sized (inversely) based on their `level` attribute.
    Node layout can be hierarchical or force-directed.
    Optionally, display debug labels near specified edges for algorithm tracing.

    Args:
        graph: A NetworkX directed graph to visualize.
        obstruction_attr: The edge attribute name containing attribute values.
        level_attr: The edge attribute name containing level values (for width).
        use_hierarchical: Whether to use hierarchical layout (True) or force-directed layout (False).
        height: Height of the visualization container.
        width: Width of the visualization container.
        bgcolor: Background color in hex format.
        font_color: Font color in hex format.
        min_edge_width: Minimum edge width for highest levels.
        max_edge_width: Maximum edge width for lowest levels.
        debug_edges: List of edge tuples (u, v) to annotate with labels.
        debug_labels: List of debug label strings corresponding to `debug_edges`.

    Returns:
        None. Opens the visualization in the default web browser.
    """
    net = _create_network(height, width, bgcolor, font_color)
    if use_hierarchical:
        _configure_hierarchical_layout(net)
    _add_nodes(net, graph)
    obs_norm, lvl_norm, cmap = _prepare_color_and_level_normalizers(graph, obstruction_attr, level_attr)

    # build debug map if annotations provided
    debug_map: dict[tuple, str] = {}
    if debug_edges is not None or debug_labels is not None:
        if not debug_edges or not debug_labels or len(debug_edges) != len(debug_labels):
            raise ValueError("`debug_edges` and `debug_labels` must both be provided and of equal length.")
        debug_map = dict(zip(debug_edges, debug_labels, strict=False))

    _add_edges(
        net,
        graph,
        obstruction_attr,
        level_attr,
        obs_norm,
        lvl_norm,
        cmap,
        min_edge_width,
        max_edge_width,
        debug_map,
    )
    _serve_network(net)


def _create_network(
    height: str,
    width: str,
    bgcolor: str,
    font_color: str,
) -> Network:
    """Create and configure a PyVis Network.

    Args:
        height: Height of the visualization container.
        width: Width of the visualization container.
        bgcolor: Background color in hex format.
        font_color: Font color in hex format.

    Returns:
        A configured PyVis Network instance.
    """
    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )
    net.toggle_physics(False)
    return net


def _configure_hierarchical_layout(net: Network) -> None:
    """Enable hierarchical layout on the PyVis Network.

    Args:
        net: The PyVis Network to configure.
    """
    net.set_options("""
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
    """)


def _prepare_color_and_level_normalizers(
    graph: nx.DiGraph,
    obstruction_attr: str,
    level_attr: str,
) -> tuple[Normalize, Normalize, LinearSegmentedColormap]:
    """Compute normalizers and colormap for edge coloring and sizing.

    Args:
        graph: A NetworkX directed graph.
        obstruction_attr: The edge attribute name containing obstruction values.
        level_attr: The edge attribute name containing level values.

    Returns:
        A tuple (obs_norm, lvl_norm, cmap) where:
            obs_norm: Normalize instance for obstruction values.
            lvl_norm: Normalize instance for level values.
            cmap: Colormap for obstruction-to-color mapping.
    """
    obs_vals = [data.get(obstruction_attr, 0.0) for _, _, data in graph.edges(data=True)]
    obs_norm = Normalize(vmin=min(obs_vals, default=0.0), vmax=max(obs_vals, default=1.0) or 1.0)
    cmap = LinearSegmentedColormap.from_list("bpr", ["#aaaaff", "#ff00ff", "#ff0000"])

    lvl_vals = [data.get(level_attr, 0.0) for _, _, data in graph.edges(data=True)]
    lvl_norm = Normalize(vmin=min(lvl_vals, default=0.0), vmax=max(lvl_vals, default=1.0) or 1.0)
    return obs_norm, lvl_norm, cmap


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
    obstruction_attr: str,
    level_attr: str,
    obs_norm: Normalize,
    lvl_norm: Normalize,
    cmap: LinearSegmentedColormap,
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
        obstruction_attr: The edge attribute name containing obstruction values.
        level_attr: The edge attribute name containing level values.
        obs_norm: Normalizer for obstruction values.
        lvl_norm: Normalizer for level values.
        cmap: Colormap for mapping normalized obstruction to RGB.
        min_edge_width: Minimum edge width for highest levels.
        max_edge_width: Maximum edge width for lowest levels.
        debug_map: Optional mapping from edge (u, v) tuples to debug label strings.
    """
    debug_map = debug_map or {}
    for u, v, data in graph.edges(data=True):
        obs = data.get(obstruction_attr, 0.0)
        r, g, b, _ = cmap(obs_norm(obs))
        color = f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"

        lvl = data.get(level_attr, 0.0)
        inv = 1.0 - lvl_norm(lvl)
        width = min_edge_width + (max_edge_width - min_edge_width) * inv

        edge_kwargs = {
            "color": color,
            "width": width,
            "title": f"{obstruction_attr}: {obs:.2f} | {level_attr}: {lvl}",
            "arrows": "to",
        }
        if (u, v) in debug_map:
            edge_kwargs["label"] = debug_map[(u, v)]
            if (u, v) in debug_map:
                edge_kwargs["label"] = debug_map[(u, v)]
                edge_kwargs["font"] = {"size": 33, "color": "#ffffff", "strokeWidth": 0, "align": "top", "vadjust": -50}
        net.add_edge(u, v, **edge_kwargs)


def _serve_network(net: Network) -> None:
    """Serve the generated PyVis HTML on a temporary local HTTP server and open it.

    Args:
        net: The PyVis Network to serve.
    """
    html_bytes = net.generate_html().encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, *args) -> None:
            pass  # silence access logs

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    host, port = server.server_address
    webbrowser.open(f"http://{host}:{port}")
    server.handle_request()
