from typing import Literal

import numpy as np
from scipy.interpolate import make_splprep

from genesis.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def plane_grid(origin: np.ndarray, normal: np.ndarray, grid_shape: int | tuple[int, int]) -> np.ndarray:
    """Compute the coordinates of a 2D plane, centered at `origin` and defined by its `normal`, in 3D space.

    Args:
        origin: (X, Y, Z) Origin coordinates.
        normal: (X, Y, Z) Vector normal to the plane.
        grid_shape: Shape of the grid to sample around the plane's origin. If a single integer M is provided,
            defines a square grid of size MxM. If a tuple (M, N) is provided, defines a rectangular grid of size MxN.

    Returns:
        (3, M, N) xyz coordinates of 2D plane grid points in the 3D space.
    """
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape, grid_shape)

    # Normalize the normal vector
    normal /= np.linalg.norm(normal)

    # Generate two orthogonal vectors to form a basis for the plane
    # using algorithm described here: https://math.stackexchange.com/a/1721127
    plane_vec = np.array([1, 0, 0])  # Arbitrary vector
    if np.allclose(plane_vec, normal, atol=1e-6):
        plane_vec = np.array([0, 1, 0])  # If normal is parallel to the arbitrary vector, use a different one
    basis_vec_1 = np.cross(normal, plane_vec)  # First basis vector orthogonal to normal
    basis_vec_1 /= np.linalg.norm(basis_vec_1)  # Normalize the first basis vector
    # Second basis vector is orthogonal to both normal and first basis vector
    # and already normalized since it is the cross product of two orthogonal normalized vectors
    basis_vec_2 = np.cross(normal, basis_vec_1)
    basis = np.stack((basis_vec_1, basis_vec_2), axis=1)  # (3, 2)

    # Create a grid of coordinates for the plane, centered around [0, 0]
    mm = np.arange(grid_shape[0]) - (grid_shape[0] // 2)  # (M,)
    nn = np.arange(grid_shape[1]) - (grid_shape[1] // 2)  # (N,)
    grid = np.stack(np.meshgrid(mm, nn, indexing="ij"))  # (2, M, N)

    # Position the grid in 3D space
    # 1) Rotate the grid around its origin by multiplying with the basis (vectorized matmul defined using einsum)
    # 2) Translate by the origin (broadcast sum with origin)
    return np.einsum("sp,pmn->smn", basis, grid) + origin[:, np.newaxis, np.newaxis]  # (3, M, N)


def smooth_curve(
    coords: np.ndarray,
    smoothing: float = 0.1,
    on_few_points: Literal["raise", "warn", "ignore"] = "raise",
) -> np.ndarray:
    """Smooth a D-dimensional curve by interpolating to subpixel coordinates using a spline.

    Args:
        coords: (D, M) Coordinates of M points on the D-dimensional curve.
        smoothing: Smoothing factor for the spline interpolation.
        on_few_points: How to handle cases where the curve has too few points to compute a spline:
            - "raise": Raise an error.
            - "warn": Log a warning, and return the original coordinates.
            - "ignore": Return the original coordinates without any warning.

    Returns:
        (D, M) Interpolated smooth curve.
    """
    if (num_points := coords.shape[1]) <= (dim := coords.shape[0]):
        msg = (
            f"Cannot smooth curve with {num_points} points in {dim}D space. "
            f"At least {dim + 1} points are required to compute a {dim}D spline."
        )
        if on_few_points == "raise":
            raise RuntimeError(msg)
        if on_few_points == "warn":
            log.warning(msg)
        elif on_few_points != "ignore":
            raise ValueError(
                f"Invalid value for `on_few_points`: {on_few_points}. Must be 'raise', 'warn', or 'ignore'."
            )

        return coords

    # Find a parametric spline representation of the curve
    spl, _ = make_splprep(coords, s=smoothing)
    # Evaluate the spline at on the range [0, 1] to get a smooth curve
    return spl(np.linspace(0, 1, num=coords.shape[1]))
