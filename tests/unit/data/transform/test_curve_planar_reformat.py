import numpy as np
import pytest

from genesis.data.transform.curve_planar_reformat import straightened_cpr

from ...synthetic_data import cylinder, torus  # noqa: TID252


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("length", [8])
def test_straightened_cpr_cylinder(radius: int, length: int) -> None:
    """Test the `straightened_cpr` function on a parametric cylinder.

    Simple test meant to help debug `straightened_cpr`, since coordinates to interpolate are manually tractable for a
    cylinder aligned with the axes.
    """
    cylinder_image, cylinder_centerline = cylinder(radius, length, centerline=True)
    straightened_cpr_reference = cylinder_image.T

    cross_section_shape = straightened_cpr_reference.shape[1:]
    cpr_image = straightened_cpr(cylinder_image, cylinder_centerline, cross_section_shape)

    # Check that the CPR image matches the expected reference image
    assert cpr_image.shape == straightened_cpr_reference.shape
    assert np.allclose(cpr_image, straightened_cpr_reference)


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("length", [20, 60])
def test_straightened_cpr_torus(radius: int, length: int) -> None:
    """Test the `straightened_cpr` function on a parametric torus."""
    torus_image, torus_centerline = torus(radius, 4 * radius, centerline_points=length)
    straightened_cpr_reference = cylinder(radius, length).T

    cross_section_shape = straightened_cpr_reference.shape[1:]
    cpr_image = straightened_cpr(torus_image, torus_centerline, cross_section_shape)

    # Check that the CPR image matches the expected reference image
    # Only check the slices that are strictly parallel/perpendicular to the X/Y axes, since their coordinates fall
    # exactly on voxels. Other slices have subpixel coordinates, and their values may not match the reference purely
    # due to interpolation.
    assert cpr_image.shape == straightened_cpr_reference.shape
    slices_to_check = np.linspace(0, length, num=4, dtype=int, endpoint=False)
    assert np.allclose(cpr_image[slices_to_check], straightened_cpr_reference[slices_to_check])
