from typing import Literal

import numpy as np
from scipy import ndimage

from genesis.utils.math import plane_grid, smooth_curve


def straightened_cpr(
    image: np.ndarray,
    tubular_centerline_coords: np.ndarray,
    cross_section_shape: int | tuple[int, int],
    centerline_smoothing: float = 0.1,
    interpolation: Literal["nearest", "bilinear"] = "nearest",
) -> np.ndarray:
    """Generates a Curve Planar Reformation (CPR) of a 3D image, based on the provided tubular centerline points.

    Args:
        image: (X, Y, Z) 3D image to be reformatted.
        tubular_centerline_coords: (3, N) xyz coordinates of the centerline points in the image.
        cross_section_shape: Shape of the cross-section to extract around the centerline points. If an integer is
             provided, then a square cross-section of that size will be generated.
        centerline_smoothing: Smoothing factor for the subpixel spline interpolation of the centerline. A higher value
            will result in a smoother centerline. Note that smoothing is only done if the centerline has at least
            4 points, i.e. ndim + 1.
        interpolation: Interpolation method to use when resampling CPR slices at subpixel coordinates.

    Returns:
        (N, *cross_section_shape) The straightened CPR of the original image, centered around the centerline.
    """
    match interpolation:
        case "nearest":
            interpolation_order = 0
        case "bilinear":
            interpolation_order = 1
        case _:
            raise ValueError(f"Interpolation method '{interpolation}' is not supported. Use 'nearest' or 'bilinear'.")

    # Determine the origin and normal of the cross-section at each centerline point
    origins = smooth_curve(tubular_centerline_coords, smoothing=centerline_smoothing, on_few_points="ignore")  # (3, N)
    # NOTE: Do not use spacing for gradients, since we want voxel gradients rather than geometrical gradients
    edge_order = 2 if origins.shape[1] > 2 else 1  # Use edge_order=2 when possible, i.e. at least 3 points, otherwise 1
    normals = np.gradient(origins, axis=-1, edge_order=edge_order)  # (3, N)

    # Compute the sub-pixel coordinates of the CPR slices centered around the centerline
    cpr_coords = [
        plane_grid(origin, normal, cross_section_shape) for origin, normal in zip(origins.T, normals.T, strict=False)
    ]  # N * (3, *cross_section_shape)
    cpr_coords = np.stack(cpr_coords, axis=1)  # (3, N, *cross_section_shape)

    # Interpolate volume at these coordinates to get the CPR image
    # Use 'grid-constant' mode so that coordinates outside the input are 0, rather than extrapolated from the input
    # (i.e. 'nearest' mode). However, due to numerical precision, coordinates may fall ever so slightly outside the
    # input (e.g. -0.0), in which case interpolation outside the input should return close to the nearest input, not 0.
    return ndimage.map_coordinates(image, cpr_coords, order=interpolation_order, mode="grid-constant", cval=0)
