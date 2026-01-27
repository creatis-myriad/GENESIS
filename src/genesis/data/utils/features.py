"""Global graph features based on node degrees, connectivity, distances, and spectral properties.

References:
    - The functions to compute the features are inspired by their implementations from the paper:
      "Extending Graph Neural Networks with Global Features" (https://openreview.net/forum?id=aisVQy6R2k)
      Code link: https://github.com/andreibrasoveanu97/gnn-global-features/blob/4eb80247885c3199d7ca39a08b49d2bc09e8e930/Scripts/compute_global_features.py
"""

import math
from collections.abc import Callable
from functools import cache, wraps
from typing import Any

import networkx as nx
from pulp import PULP_CBC_CMD, LpBinary, LpMaximize, LpProblem, LpVariable, lpSum


def undirected_graph(as_view: bool = True) -> Callable:
    """Decorator to make sure that the graph forwarded to the decorated function is undirected.

    Args:
        as_view: If `True`, returns an undirected view of the original directed graph. Otherwise, returns a deep copy
            of the original directed graph.

    Returns:
        Function decorated to ensure graph argument is undirected.
    """

    def undirected_decorator(func: Callable[[nx.Graph, ...], ...]) -> Callable[[nx.Graph, ...], ...]:
        @wraps(func)
        def undirected_graph_fn(graph: nx.Graph, *args, **kwargs) -> Any:
            return func(graph.to_undirected(as_view=as_view), *args, **kwargs)

        return undirected_graph_fn

    return undirected_decorator


def wiener_index(graph: nx.Graph) -> int:
    """Sum of the lengths of the shortest paths between all pairs of vertices.

    Wraps NetworkX's `wiener_index` function to return -1 (instead of infinity) for disconnected graphs.

    Examples:
        The Wiener index of the (unweighted) complete graph on *n* nodes equals the number of pairs of the *n* nodes,
        since each pair of nodes is at distance one.
        >>> n = 10
        >>> G = nx.complete_graph(n)
        >>> wiener_index(G) # n * (n - 1) / 2 = 10 * 9 / 2 = 45
        45

        Graphs that are not strongly-connected now return -1 (instead of the Wiener index's definition of infinity).
        >>> G = nx.empty_graph(2)
        >>> wiener_index(G)
        -1
    """
    return -1 if math.isinf(val := nx.wiener_index(graph)) else int(val)


@undirected_graph()
def circuit_rank(graph: nx.Graph) -> int:
    """Minimum number of edges that must be removed from the graph to break all its cycles.

    Also known as first Betti number.

    Examples:
        See: https://en.wikipedia.org/wiki/Cyclomatic_number
        >>> g = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 1), (5, 2)])
        >>> circuit_rank(g)
        2
    """
    return graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)


@undirected_graph()
def independence_number(graph: nx.Graph) -> int:
    """Cardinality of the largest node set in a graph, such that no two nodes in this set are connected by an edge.

    References:
        - A function to compute the independence number exists in the `grinpy` package, which extends NetworkX.
          However, `grinpy` is not maintained anymore, and has a circular import bug preventing its use.
          (GitHub issue: https://github.com/somacdivad/grinpy/issues/30)
          Thus, we adapted `grinpy`'s integer linear programming method to solve the largest independent set problem,
          which relies on the maintained `pulp` package rather than internal `grinpy` functions. Implementation link:
          https://github.com/somacdivad/grinpy/blob/597f9109b84f1c1aa8c8dd2ac5b572a05ba474de/grinpy/invariants/independence.py#L168-L215

    Examples:
        A path graph with 5 vertices has 4 edges: (0, 1), (1, 2), (2, 3), (3, 4).
        The largest independent set is nodes {0, 2, 4}, with a size of 3.
        >>> p5 = nx.path_graph(5)
        >>> independence_number(p5)
        3
    """
    prob = LpProblem("min_total_dominating_set", LpMaximize)
    variables = {node: LpVariable(f"x{i + 1}", 0, 1, LpBinary) for i, node in enumerate(graph.nodes())}

    # Set the domination number objective function
    prob += lpSum(variables)

    # Set constraints for independence
    for e in graph.edges():
        prob += variables[e[0]] + variables[e[1]] <= 1

    # Instantiate solver manually, instead of relying on default solver selection, to disable solver log output
    prob.solve(PULP_CBC_CMD(msg=False))
    solution_set = {node for node in variables if variables[node].value() == 1}
    return len(solution_set)


@undirected_graph()
def maximum_matching(graph: nx.Graph) -> int:
    """Cardinality of the maximum matching in the graph, i.e. subset of edges in which no node occurs more than once.

    Examples:
        >>> g = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
        >>> maximum_matching(g) # maximum matching is set of edges {(1, 2), (3, 5)} with a size of 2
        2
    """
    return len(nx.maximal_matching(graph))


def second_eigenval(graph: nx.Graph) -> float:
    """Value of the second-smallest eigenvalue of matrix reflects how well connected the graph is.

    Smaller values correspond to less connected graphs, while for fully connected graphs, the value is relatively
    larger.
    """
    return nx.laplacian_spectrum(graph)[1]


def spectral_radius(graph: nx.Graph) -> float | complex:
    """Largest absolute value of the graph spectrum, i.e. eigenvalues of the adjacency matrix."""
    return max(nx.adjacency_spectrum(graph))


def zagreb_index1(graph: nx.Graph) -> int:
    """Sum of squares of the degrees of the vertices.

    Examples:
        See: https://en.wikipedia.org/wiki/Zagreb_indices
        >>> g = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (2, 4)])
        >>> zagreb_index1(g) # degrees are [4, 2, 3, 2, 3], so sum of squares is 4^2 + 2^2 + 3^2 + 2^2 + 3^2 = 42
        42
    """
    return sum(degree**2 for _, degree in graph.degree)


def zagreb_index2(graph: nx.Graph) -> int:
    """Sum of the products of the degrees of pairs of adjacent nodes.

    Examples:
        See: https://en.wikipedia.org/wiki/Zagreb_indices
        >>> g = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (2, 4)])
        >>> zagreb_index2(g)
        61
    """
    return sum(graph.degree(u) * graph.degree(v) for u, v in graph.edges)


@undirected_graph()
def total_matchings(graph: nx.Graph) -> int:
    """Total number of matchings in a graph, i.e. subsets of edges in which no node occurs more than once.

    This implementation is optimized by using memoized recursion. Nevertheless, it's only suitable for relatively small
    graphs, due to its exponential time/space complexity w.r.t. the number of edges.

    Examples:
        A complete graph with 4 vertices has 9 non-empty matchings, plus the empty matching.
        See: https://en.wikipedia.org/wiki/Hosoya_index#/media/File:K4_matchings.svg
        >>> n = 4
        >>> G = nx.complete_graph(4)
        >>> total_matchings(G)
        10

        A path graph with 4 vertices has 3 edges: (0, 1), (1, 2), (2, 3).
        Matchings of size 0: 1 (the empty set)
        Matchings of size 1: 3 ({(0, 1)}, {(1, 2)}, {(2, 3)})
        Matchings of size 2: 1 ({(0, 1), (2, 3)})
        Total Matchings: 1 + 3 + 1 = 5
        >>> p4 = nx.path_graph(4)
        >>> total_matchings(p4)
        5

        A complete graph with 3 vertices has 3 edges.
        Matchings of size 0: 1
        Matchings of size 1: 3
        Matchings of size 2: 0
        Total Matchings: 1 + 3 = 4
        >>> f3 = nx.complete_graph(3)
        >>> total_matchings(f3)
        4
    """
    # Convert the graph's edges into a hashable key (nested sorted tuples) uniquely identifying the subgraph.
    # Sort nodes within each edge tuple to handle (u, v) vs (v, u) consistency,
    # and sort the list of edges to ensure consistent ordering.
    edge_list = tuple(sorted(tuple(sorted(e)) for e in graph.edges()))

    # Define the recursive logic to memoize in an inner function wrapped by `functools.cache`.
    # The cache applies to the hashable 'edges_key' argument.
    @cache
    def recurse(edges_key: tuple[tuple[int, int], ...]) -> int:
        # 1. Base Case
        if not edges_key:
            return 1

        # 2. Pick an arbitrary edge (u, v) from the key
        # Since the key is sorted, the first element is a canonical choice.
        u, v = edges_key[0]

        # --- Case 1: Matchings that DO NOT use the edge (u, v) (G - e) ---
        # Key for G - e is simply the rest of the edges
        edges_without_e = edges_key[1:]
        count_without_e = recurse(edges_without_e)

        # --- Case 2: Matchings that DO use the edge (u, v) (G - V(e)) ---
        # Remove all edges incident to u or v from the current edges_key
        edges_with_e = tuple(edge for edge in edges_key[1:] if not (u in edge or v in edge))
        count_with_e = recurse(edges_with_e)

        # 3. Sum the results
        return count_without_e + count_with_e

    # Start the recursion with the initial graph's key
    return recurse(edge_list)
