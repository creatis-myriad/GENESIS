import math

import numpy as np
from scipy import ndimage

from genesis.data.transform.curve_planar_reformat import straightened_cpr
from genesis.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def transversal_obstruction(
    vessel_mask: np.ndarray,
    obstruction_mask: np.ndarray,
    vessel_centerline: np.ndarray,
    cpr_padding: int | float = 1.0,
) -> np.ndarray:
    """Computes the transversal obstruction of a vessel in a 3D image, for each point along the vessel centerline.

    Args:
        vessel_mask: (X, Y, Z) 3D segmentation of the vessels. Any non-zero voxel is considered part of the mask.
        obstruction_mask: (X, Y, Z) 3D segmentation of the structures (e.g. thrombi) to consider as obstructions in the
            vessels. Any non-zero voxel is considered part of the mask.
        vessel_centerline: (3, N) xyz coordinates of the centerline points of the vessel segment to analyze.
        cpr_padding: Padding to add to the detected radius of the vessel to compute the size of cross-sections in the
            Curve Planar Reformation (CPR) representation. If an integer, it is the number of pixels to pad; if a float,
            it is the fraction of the measured radius to use as padding.

    Returns:
        (N) The transversal obstruction values at each point along the vessel centerline.
    """
    # Combine the vessels and obstructions in a single mask, labeled with 1 for vessels and 2 for obstructions.
    # The obstructions are overlaid on top of the vessels
    _vessel_label = 1
    _obstruction_label = 2
    vessel_and_obstruction_mask = vessel_mask.astype(bool) * _vessel_label
    vessel_and_obstruction_mask[obstruction_mask.astype(bool)] = _obstruction_label

    # Heuristic for the size of the ROI around the centerline points:
    # Take the maximum distance between a centerline point and the background as the radius of the ROI
    dist_to_bg = ndimage.distance_transform_edt(vessel_and_obstruction_mask)
    centerline_dist_to_bg = dist_to_bg[*vessel_centerline]
    root_radius = math.ceil(np.max(centerline_dist_to_bg))  # Round up the max radius to an integer
    match cpr_padding:
        case int():
            root_radius += cpr_padding  # Add some padding to ensure the ROI is large enough
        case float():
            root_radius += math.ceil(cpr_padding * root_radius)  # Add a fraction of the radius as padding
        case _:
            raise ValueError(f"Invalid type for `cpr_padding`: {type(cpr_padding)}. Expected int or float.")

    # Compute the Curve Planar Reformation (CPR) representation of the vessel segment
    vessel_cpr = straightened_cpr(vessel_and_obstruction_mask, vessel_centerline, 2 * root_radius)

    # Ignore components disconnected from the centerline component, e.g. other vessels crossing the CPR's ROI,
    # to avoid having them interfere with the obstruction metric
    vessel_cpr = _remove_non_centerline_structures(vessel_cpr)

    # Check that remaining vessel/obstruction voxels are not on the ROI's edges, to make sure they are not cropped
    # In practice, implement this by checking that cropping 1 voxel from each side of the ROI does not alter the mask
    # This sanity check is only performed at the first slice of the CPR (i.e. the root of the centerline), because in
    # later slices child vessels might inevitably branch out to the edge of the ROI.
    cpr_root_slice_crop = vessel_cpr[0, 1:-1, 1:-1]
    if np.sum(cpr_root_slice_crop) != np.sum(vessel_cpr[0]):
        log.warning(
            "The Curve Planar Reformation (CPR) vessel representation contains vessel/obstruction voxels at the edges, "
            "indicating that part of the vessel/obstructions might be cropped out. "
            "You should increase `cpr_padding` to ensure the entire segment and obstruction are included."
        )

    # Compute the total vessel lumen to obstruction ratio for each point along the centerline
    vessel_area = np.sum(vessel_cpr == _vessel_label, axis=(1, 2))
    obstruction_area = np.sum(vessel_cpr == _obstruction_label, axis=(1, 2))
    return obstruction_area / (vessel_area + obstruction_area)


def _remove_non_centerline_structures(cpr: np.ndarray) -> np.ndarray:
    """Filters out structures apart from the centerline around which the CPR was computed.

    Args:
        cpr: (N, X, Y) CPR image, where N is the number of cross-section of the structure of interest, and X, Y are the
            dimensions of each cross-section.

    Returns:
        (N, X, Y) CPR image with only the centerline structure retained, and all other structures removed.
    """
    # Create a 3D connectivity structure that ignores 3D connectivity between slices
    intraslice_connectivity = np.stack([np.zeros((3, 3)), ndimage.generate_binary_structure(2, 1), np.zeros((3, 3))])
    # Label connected components in each slice individually
    labels, _ = ndimage.label(cpr, structure=intraslice_connectivity)
    # Look at each slice's connectivity individually, since components that separate downstream might be connected
    # near the root of the centerline, and thus 3D connectivity would not remove them
    cpr_x_mid, cpr_y_mid = np.array(cpr.shape[1:]) // 2
    centerline_labels = labels[:, cpr_x_mid, cpr_y_mid]  # Labels of the components at the center of each CPR slice
    return cpr * (centerline_labels == labels.T).T  # Ignore components disconnected from the center one
