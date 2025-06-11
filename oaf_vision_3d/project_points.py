# %% [markdown]
# # Project Points
#
# This function projects 3D points into a 2D image using a
# [`LensModel`](lens_model.py) object. The process for this was discussed in more
# detail in the workshop [4: 3D-2D Projections and PnP](../workshops/04_3d_2d_projections_and_pnp.ipynb).

# %%
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def project_points(
    points: NDArray[Shape["*, 3"], Float32],
    lens_model: LensModel,
    transformation_matrix: TransformationMatrix = TransformationMatrix(),
) -> NDArray[Shape["*, 2"], Float32]:
    transformed_points = transformation_matrix @ points[None, ...]

    undistorted_normalized_pixels = (
        transformed_points[..., :2] / transformed_points[..., 2:]
    )
    normalized_pixels = lens_model.distort_pixels(
        normalized_pixels=undistorted_normalized_pixels
    )
    return lens_model.denormalize_pixels(pixels=normalized_pixels)[0]
