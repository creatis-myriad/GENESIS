import logging

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture

from genesis.metrics.vessel import transversal_obstruction

from ..synthetic_data import cylinder  # noqa: TID252


@pytest.mark.parametrize("radius", [2, 4])
@pytest.mark.parametrize("length", [5, 10])
@pytest.mark.parametrize("obstruction", [0.5, 1.0])
def test_transversal_obstruction(radius: int, length: int, obstruction: float) -> None:
    """Test the `transversal_obstruction` function on a parametric cylinder."""
    # Generate a synthetic cylinder with a given radius and length
    cylinder_mask, cylinder_centerline = cylinder(radius, length, centerline=True)
    cross_section = cylinder_mask[..., 0]  # Take the first cross-section of the cylinder

    # Compute the absolute size of the obstruction to generate in the cylinder's first cross-section
    cylinder_cross_section_size = np.sum(cross_section)
    obstruction_size = int(cylinder_cross_section_size * obstruction)

    # Create an obstruction mask with the first N pixels (N = `obstruction_size`) of the cylinder's first cross-section
    cylinder_coords = np.stack(np.nonzero(cross_section))
    obstruction_coords = cylinder_coords[:, :obstruction_size]
    obstruction_mask = np.zeros_like(cylinder_mask)
    obstruction_mask[*obstruction_coords, 0] = 1

    # Check that the first slice of the cylinder has the obstruction, while the remaining slices are unobstructed
    # Since the discrete obstruction might not be able to approximate the target obstruction ratio exactly,
    # we add a tolerance corresponding to the relative size of one voxel in the cross-section
    measured_obstruction = transversal_obstruction(cylinder_mask, obstruction_mask, cylinder_centerline)
    assert len(measured_obstruction) == length
    assert np.allclose(measured_obstruction[0], obstruction, atol=1 / cylinder_cross_section_size)
    assert np.allclose(measured_obstruction[1:], 0)


@pytest.mark.parametrize("radius", [4])
@pytest.mark.parametrize("length", [10])
def test_transversal_obstruction_disconnected_vessel(radius: int, length: int) -> None:
    """Test the `transversal_obstruction` function on a parametric cylinder split into two disconnected components."""
    # Generate a synthetic cylinder with a given radius and length, and an empty obstruction mask
    cylinder_mask, cylinder_centerline = cylinder(radius, length, centerline=True)
    obstruction_mask = np.zeros_like(cylinder_mask)

    # Create a disconnection in the cylinder mask
    disconnected_cylinder_mask = cylinder_mask.copy()
    mid_slice = length // 2
    disconnected_cylinder_mask[..., mid_slice] = 0  # Remove the middle slice to create a disconnection

    # Check that the transversal obstruction is NaN for the slice without vessel or obstruction but 0 for the others
    measure_obstruction = transversal_obstruction(disconnected_cylinder_mask, obstruction_mask, cylinder_centerline)
    assert len(measure_obstruction) == length
    assert np.allclose(measure_obstruction[:mid_slice], 0)
    assert np.allclose(measure_obstruction[mid_slice + 1 :], 0)
    assert np.isnan(measure_obstruction[mid_slice])


@pytest.mark.parametrize("radius", [4])
@pytest.mark.parametrize("length", [10])
def test_transversal_obstruction_disconnected_component(radius: int, length: int) -> None:
    """Test the `transversal_obstruction` function on a parametric cylinder with a disconnected obstruction."""
    # Generate a synthetic cylinder with a given radius and length, and an empty obstruction mask
    cylinder_mask, cylinder_centerline = cylinder(radius, length, centerline=True)
    obstruction_mask = np.zeros_like(cylinder_mask)

    # Add a small disconnected component outside the cylinder
    disconnected_obstruction_mask = cylinder_mask.copy()
    disconnected_obstruction_mask[0, 0, 0] = 1  # Add a small disconnected component

    # Check that the disconnected obstruction component was removed, and did not count towards the vessel's obstruction
    measure_obstruction = transversal_obstruction(disconnected_obstruction_mask, obstruction_mask, cylinder_centerline)
    assert len(measure_obstruction) == length
    assert np.allclose(measure_obstruction, 0)


@pytest.mark.parametrize("radius", [4])
@pytest.mark.parametrize("length", [5])
def test_transversal_obstruction_edge_mask(
    caplog: LogCaptureFixture, mocker: MockerFixture, radius: int, length: int
) -> None:
    """Test that `transversal_obstruction` warns on a parametric cylinder with vessel voxels at the edges."""
    # Generate a synthetic cylinder with a given radius and length, and an empty obstruction mask
    cylinder_mask, cylinder_centerline = cylinder(radius, length, centerline=True)
    obstruction_mask = np.zeros_like(cylinder_mask)

    # Mock `straightened_cpr` to return an array of 1s (so that the structure touches the edges) as the CPR image
    mocker.patch("genesis.metrics.vessel.straightened_cpr", return_value=np.ones_like(cylinder_mask.T))

    # Check that the function logs a warning due to the voxels of interest at the edges of the CPR representation
    with caplog.at_level(logging.INFO):
        transversal_obstruction(cylinder_mask, obstruction_mask, cylinder_centerline)
    assert "(CPR) vessel representation contains vessel/obstruction voxels at the edges" in caplog.text
