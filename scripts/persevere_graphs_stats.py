from statistics import mean

import rootutils

from genesis.data.persevere import PersevereDataset

dataset_root = rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE"
dataset = PersevereDataset(
    root=str(dataset_root),
    line_graph=False,  # Use original graph topology for statistics
)
graphs_records = [
    {
        "n_nodes": data.num_nodes,
        "n_edges": data.num_edges,
        "avg_node_degree": 2 * data.num_edges / data.num_nodes,
    }
    for data in dataset
]
dataset_stats = {
    "n_graphs": len(dataset),
    "avg_n_nodes": mean(record["n_nodes"] for record in graphs_records),
    "avg_n_edges": mean(record["n_edges"] for record in graphs_records),
    "avg_node_degree": mean(record["avg_node_degree"] for record in graphs_records),
}
print("PERSEVERE Dataset Statistics:")
print(dataset_stats)
