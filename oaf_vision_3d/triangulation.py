# %% [markdown]
# # Triangulation
#
# This function triangulates 3D points from two sets of two undistorted normalized
# pixels and a [`TransformationMatrix`](transformation_matrix.py) object. The process
# for this was discussed in more detail in the workshop
# [5: Dual Camera Setups](../workshops/05_dual_camera_setups.ipynb).


# %%
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def triangulate_points(  # type: ignore
    undistorted_normalized_pixels_0: NDArray[Shape["H, W, 2"], Float32],
    undistorted_normalized_pixels_1: NDArray[Shape["H, W, 2"], Float32],
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]: ...


def triangulate_disparity(
    disparity: NDArray[Shape["H, W"], Float32],
    lens_model_0: LensModel,
    lens_model_1: LensModel,
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]:
    y, x = np.indices(disparity.shape, dtype=np.float32)
    pixels_0 = np.stack([x, y], axis=-1)
    pixels_1 = np.stack([x - disparity, y], axis=-1)

    undistortied_normalized_pixels_0 = lens_model_0.undistort_pixels(
        normalized_pixels=lens_model_0.normalize_pixels(pixels=pixels_0)
    )
    undistortied_normalized_pixels_1 = lens_model_1.undistort_pixels(
        normalized_pixels=lens_model_1.normalize_pixels(pixels=pixels_1)
    )

    return triangulate_points(
        undistorted_normalized_pixels_0=undistortied_normalized_pixels_0,
        undistorted_normalized_pixels_1=undistortied_normalized_pixels_1,
        transformation_matrix=transformation_matrix,
    )
