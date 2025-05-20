import numpy as np
import skimage as ski


def cylinder(radius: int, length: int, centerline: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate a 3D image of a parametric cylinder, with cross-section in XY plane and length along Z axis.

    Args:
        radius: The radius of the cross-section of the cylinder.
        length: The length of the cylinder.
        centerline: If True, also return the centerline points of the cylinder.

    Returns:
        (2 * `radius` + 1, 2 * `radius` + 1, `length`) The 3D numpy array representing the cylinder.
        If `centerline` is True:
            (3, `length`) xyz coordinates of points on the cylinder's centerline.
    """
    # Generate the 3D cylinder image by stacking 2D cross-sections
    cross_section = ski.morphology.disk(radius)
    cylinder_volume = np.stack([cross_section] * length, axis=-1)

    if not centerline:
        return cylinder_volume

    # Compute the centerline points of the cylinder
    cross_section_center_coords = np.array(cross_section.shape) // 2
    centerline_xy_coords = np.tile(cross_section_center_coords[:, np.newaxis], (1, length))
    centerline_z_coords = np.arange(length)
    centerline = np.vstack((centerline_xy_coords, centerline_z_coords))

    return cylinder_volume, centerline


def torus(
    crosssection_radius: int, ring_radius: int, centerline_points: int | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate a cube image of a parametric torus.

    Args:
        crosssection_radius: The radius of the cross-section of ring of the torus.
        ring_radius: The radius from the center to the middle of the ring of the torus.
        centerline_points: If None or 0, only the Torus is returned. If > 0, also return this number of points sampled
            along the centerline of the torus.

    Returns:
        (X, Y, Z) The 3D numpy array representing the torus.
        If `centerline_points` is None or 0:
            (3, `centerline_points`) xyz coordinates of points on the torus' centerline.
    """
    # Define other parameters w.r.t. the desired tube radius
    margin_pad = 1
    volume_radius = ring_radius + crosssection_radius + margin_pad

    # Create a 3D grid of points
    start, stop = -volume_radius, volume_radius + 1
    grid = np.mgrid[start:stop, start:stop, start:stop]  # (3, X, Y, Z)
    x2, y2, z2 = np.square(grid)  # 3 * (X, Y, Z)

    # Compute the equation of the torus on the 3D grid, to get a continuous representation
    r2 = np.square(ring_radius - np.sqrt(x2 + y2)) + z2
    torus_volume = np.exp(-r2 / (crosssection_radius * crosssection_radius))

    # Threshold the torus to create a binary occupancy mask
    torus_volume = (torus_volume > 0.35).astype(np.uint8)

    if not centerline_points:
        return torus_volume

    # Use negative angle of rotation so that rotation appears clockwise in classic XY plane visualization, where
    # coordinates 0,0 are in the top left corner
    theta = np.linspace(0, -2 * np.pi, centerline_points, endpoint=False)
    centerline_x = (ring_radius * np.cos(theta)) + volume_radius
    centerline_y = (ring_radius * np.sin(theta)) + volume_radius
    centerline = np.vstack((centerline_x, centerline_y, np.full(centerline_points, volume_radius)))
    return torus_volume, centerline
