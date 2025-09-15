from collections.abc import Callable
from pathlib import Path
from typing import Literal

import click
import numpy as np
import rootutils
from click import pass_obj
from tqdm.auto import tqdm

from genesis.analysis.cli.utils import get_logger
from genesis.analysis.plot.graph import networkx_to_pyvis
from genesis.analysis.scores.mastora import mastora as mastora_score
from genesis.analysis.scores.qanadli import qanadli as qanadli_score
from genesis.data.utils import find_file, load_nifti, networkx_has_attributes

log = get_logger(__name__)

# Options common to scoring commands
option_obstruction_attr = click.option(
    "--obstruction-attr",
    "-oa",
    type=str,
    default="transversal_obstruction_max",
    show_default=True,
    help="Edge attribute to use as obstruction values.",
)


@click.command()
@option_obstruction_attr
@click.option(
    "--partial-obstruction-thresh",
    "-po",
    type=float,
    default=0.25,
    show_default=True,
    help="Transversal obstruction threshold to consider a segment partially obstructed.",
)
@click.option(
    "--total-obstruction-thresh",
    "-to",
    type=float,
    default=0.75,
    show_default=True,
    help="Transversal obstruction threshold to consider a segment totally obstructed.",
)
@click.pass_obj
def qanadli(obj: dict, *args, **kwargs) -> None:
    """Compute Qanadli score on the graph(s).

    Computes the Qanadli score for pulmonary embolism risk assessment, considering both embolus location and degree of
    obstruction, weighting each segment by its number of distal subsegments.
    """
    _run_graph_obstruction_score(obj, qanadli_score, *args, **kwargs)


@click.command()
@click.argument(
    "mode",
    type=click.Choice(["central", "peripheral", "global"]),
)
@option_obstruction_attr
@click.pass_obj
def mastora(obj: dict, mode: Literal["central", "peripheral", "global"], **kwargs) -> None:  # noqa: D417
    """Compute Mastora score on the graph(s).

    Computes the Mastora score for pulmonary embolism risk assessment, evaluating the precise degree of vascular
    obstruction in central and/or peripheral arteries.

    Args:
        mode: Variant of the Mastora score to compute.
            - 'central': Considers obstructions in the mediastinal and lobar arteries
            - 'peripheral': Considers obstructions in the segmental arteries
            - 'global': Considers obstructions in all arteries (i.e. both central and peripheral)
    """
    _run_graph_obstruction_score(obj, mastora_score, mode, score_name=f"mastora_{mode}", **kwargs)


def _run_graph_obstruction_score(
    obj: dict,
    score_fn: Callable,
    *score_args,
    score_name: str | None = None,
    obstruction_attr: str = "transversal_obstruction_max",
    **score_kwargs,
) -> None:
    """Compute a global score on graph(s) (previously loaded in the click context), with optional debug visualization.

    Args:
        obj: State dict to store objects and communicate between commands, part of the Click context.
        score_fn: The global obstruction score function.
        *score_args: Positional arguments to pass along to `score_fn`.
        score_name: Name to use for the score in logging and storing in `obj`. If None, uses `score_fn.__name__`.
        obstruction_attr: Edge attribute to use as obstruction values.
        **score_kwargs: Additional parameters to pass along to `score_fn`.
    """
    score_name = score_name or score_fn.__name__

    graphs = obj["graphs"]
    scores = {}
    debug_info = {}
    for patient_id, graph in tqdm(graphs.items(), desc=f"Compute {score_name} score on input graphs", unit="graph"):
        score, graph_debug_info = score_fn(
            graph, *score_args, obstruction_attr=obstruction_attr, debug=True, **score_kwargs
        )
        scores[patient_id] = score
        debug_info[patient_id] = graph_debug_info

    obj[score_name] = {
        "values": scores,
        "debug_info": debug_info,
        "obstruction_attr": obstruction_attr,
    }
    # If only one graph was processed, show its score
    if len(scores) == 1:
        patient_id = next(iter(scores))
        log.info(f"{score_name} score for patient {patient_id}: {scores[patient_id]}")


@click.command()
@click.option(
    "--annotations-ventricles-dir",
    "-d",
    type=click.Path(file_okay=False, path_type=Path),
    default=rootutils.find_root(indicator="pyproject.toml") / "data/PERSEVERE/annotations_ventricles",
    show_default=True,
    help="Directory containing the right and left ventricles masks to compute ratio from. "
    "If empty, masks will be inferred using TotalSegmentator and stored here for future use.",
)
@click.option(
    "--mask-pattern",
    "-m",
    type=str,
    default="*{id}*.nii.gz",
    show_default=False,
    help="Glob pattern to search for segmentation masks within `annotations-ventricles-dir`.",
)
@click.option(
    "--lv-label",
    "-lv",
    type=int,
    default=3,
    show_default=True,
    help="Label value for the left ventricle in the masks. "
    "Default to the label used by TotalSegmentator with the `heartchambers_highres` task.",
)
@click.option(
    "--rv-label",
    "-rv",
    type=int,
    default=5,
    show_default=True,
    help="Label value for the right ventricle in the masks. "
    "Default to the label used by TotalSegmentator with the `heartchambers_highres` task.",
)
@click.pass_obj
def rv_lv_ratio(obj: dict, annotations_ventricles_dir: Path, mask_pattern: str, lv_label: int, rv_label: int) -> None:
    """Compute the right-over-left ventricle volume ratio on the CTPA image(s).

    If no pre-computed masks are provided, they will be inferred using TotalSegmentator and saved for future use. This
    requires TotalSegmentator to be installed if no pre-computed masks are available. It also means the first run may
    take a while, up to several hours depending on the number of images and your hardware.
    """
    if not (ctpa_files := obj.get("ctpa_paths")):
        raise ValueError(
            "No CTPA files found in the context. Please load them using by setting `--ctpa-pattern` to look for under "
            "`--search-dir`."
        )

    if not annotations_ventricles_dir.exists():
        annotations_ventricles_dir.mkdir(parents=True)
        log.info(f"Created annotations directory at {annotations_ventricles_dir} save inferred masks for future use.")

    rv_volumes = {}
    lv_volumes = {}
    rv_lv_ratios = {}
    for patient_id, ctpa_file in tqdm(ctpa_files.items(), desc="Compute RV/LV ratio on input CTPA", unit="image"):
        try:
            mask_path = find_file(patient_id, search_dirs=[annotations_ventricles_dir], pattern=mask_pattern)
        except FileNotFoundError as e:
            if "No file found" in str(e):
                # If no pre-computed mask is found, compute it now and save it for future use
                try:
                    from totalsegmentator.python_api import totalsegmentator  # noqa: PLC0415
                except ImportError:
                    raise ImportError(
                        "No pre-computed ventricle mask found and TotalSegmentator is not installed. Please provide "
                        "pre-computed masks or install TotalSegmentator to enable predicting masks automatically."
                    ) from None

                mask_path = annotations_ventricles_dir / f"{patient_id}.nii.gz"
                totalsegmentator(ctpa_file, mask_path, task="heartchambers_highres", ml=True)

            elif "Multiple matches" in str(e):
                # Log and skip if multiple files are found
                log.exception("", exc_info=True)
                continue
            else:
                # Re-raise unexpected errors
                raise

        # Load the mask and metadata
        img, metadata = load_nifti(mask_path)

        # Compute voxel volume from spacing
        spacing = metadata["pixdim"][1:4]  # Voxel spacing (in mm)
        voxel_volume = np.prod(spacing) / 1000  # Convert mm^3 to mL

        # Compute the volume from the masks and CTPA spacing
        rv_volume = np.sum(img == rv_label) * voxel_volume
        lv_volume = np.sum(img == lv_label) * voxel_volume

        rv_volumes[patient_id] = rv_volume
        lv_volumes[patient_id] = lv_volume
        rv_lv_ratios[patient_id] = rv_volume / lv_volume

    obj["rv_volume"] = {"values": rv_volumes}
    obj["lv_volume"] = {"values": lv_volumes}
    obj["rv_lv_ratio"] = {"values": rv_lv_ratios}

    # If only one graph was processed, show its measures
    if len(ctpa_files) == 1:
        patient_id = next(iter(ctpa_files))
        log.info(
            f"Ventricle volumes (in mL) for patient {patient_id}: RV={rv_volumes[patient_id]}, "
            f"LV={lv_volumes[patient_id]}, ratio={rv_lv_ratios[patient_id]}"
        )


@click.command()
@click.option(
    "--color-attr",
    "-c",
    type=str,
    help="Edge attribute to color edges in the graph visualization. Ignored in favor of obstruction attribute used "
    "by score if `--debug-score` is also specified.",
)
@click.option(
    "--debug-score",
    "-d",
    type=click.Choice(["qanadli", "mastora_central", "mastora_peripheral", "mastora_global"]),
    help="Score for which to display intermediate values in the visualization, to help debugging.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Directory to save the generated HTML files under.",
)
@pass_obj
def visualize(
    obj: dict,
    color_attr: str | None = None,
    debug_score: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Visualize attribute values in the graph(s) using an interactive PyVis-generated HTML.

    Creates an interactive network visualization of the arterial tree, coloring edges by the specified attribute.
    """
    if score_data := obj.get(debug_score):
        if color_attr:
            log.warning(
                f"Both `--color-attr` and `--debug-score` are specified. '{color_attr}' value "
                f"(`--color-attr`) ignored in favor of '{score_data['obstruction_attr']}' attribute used by "
                f"{debug_score} (`--debug-score`)."
            )
        color_attr = score_data["obstruction_attr"]

    output_dir = output_dir or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    for patient_id, graph in tqdm(
        obj["graphs"].items(), desc="Generating interactive visualizations for input graphs", unit="graph"
    ):
        if color_attr and not networkx_has_attributes(graph, element="edges", attrs=[color_attr]):
            raise ValueError(
                f"Graph of patient {patient_id} does not have edge attribute '{color_attr}' required for visualization."
            )

        visu_filename = f"{patient_id}.html"
        debug_info = None
        if debug_score:
            debug_info = score_data["debug_info"][patient_id]
            visu_filename = f"{patient_id}_{color_attr}_{debug_score}.html"

        net = networkx_to_pyvis(graph, color_attr=color_attr, debug_info=debug_info)
        net.show(str(output_dir / visu_filename), notebook=False)
