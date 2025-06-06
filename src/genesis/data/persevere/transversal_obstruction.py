def main() -> None:
    """Main script."""
    import argparse
    from pathlib import Path

    from genesis.data.utils import json_to_networkx, load_nifti, networkx_to_json
    from genesis.metrics.vessel import full_graph_transversal_obstructions
    from genesis.utils.path import append

    parser = argparse.ArgumentParser(
        description="Compute the transversal obstruction on each segment for a given segmentation and vessel graph."
    )
    parser.add_argument("vessel_mask", type=Path, help="Path to the vessel segmentation mask.")
    parser.add_argument("obstruction_mask", type=Path, help="Path to the obstruction segmentation mask.")
    parser.add_argument("vessel_graph", type=Path, help="Path to the vessel graph in NetworkX's JSON format.")
    parser.add_argument(
        "--centerline_key", type=str, default="centerline", help="Key of the centerline in the vessel graph edges."
    )
    parser.add_argument(
        "--obstruction_key",
        type=str,
        default="transversal_obstruction",
        help="Key under which to save the transversal obstruction in the vessel graph edges.",
    )
    parser.add_argument(
        "--cpr_padding",
        type=float,
        default=1.0,
        help="Padding to add around vessels to make sure to include them in the ROI when computing obstructions.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=Path,
        help="Path to the output file where to copy the graph with the added transversal obstruction information.",
    )
    args = parser.parse_args()

    # Load the vessel and obstruction masks
    vessel_mask = load_nifti(args.vessel_mask)[0]
    obstruction_mask = load_nifti(args.obstruction_mask)[0]

    # Load the vessel graph
    vessel_graph = json_to_networkx(args.vessel_graph, edges="links")
    # Compute the transversal obstruction
    vessel_graph_with_obstructions = full_graph_transversal_obstructions(
        vessel_mask,
        obstruction_mask,
        vessel_graph,
        centerline_key=args.centerline_key,
        obstruction_key=args.obstruction_key,
        cpr_padding=args.cpr_padding,
        progress_bar=True,
    )

    # If no output file is specified, default to the input file with a suffix
    if not (output_file := args.output_file):
        output_file = append(args.vessel_graph, f"_{args.obstruction_key}")
    # Save the graph with the added transversal obstruction information
    networkx_to_json(vessel_graph_with_obstructions, output_file)


if __name__ == "__main__":
    main()
