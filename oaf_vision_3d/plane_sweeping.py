# %% [markdown]
# # Plane Sweeping
#
# This function performs plane sweeping to estimate the depth of a pixel in a set of 2D
# images. The process for this was discussed in more detail in the workshop
# [7: Stereo Matching Fundamentals Continues](../workshops/07_stereo_matching_fundamentals_continued.ipynb).


# %%

import numpy as np
from nptyping import Float32, NDArray, Shape
from scipy.ndimage import map_coordinates

from oaf_vision_3d.lens_model import LensModel
from oaf_vision_3d.project_points import project_points
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def repeoject_image_at_depth(
    image: NDArray[Shape["H, W, ..."], Float32],
    camera_vectors: NDArray[Shape["H, W, 3"], Float32],
    depth: float,
    lens_model: LensModel,
    transformation_matrix: TransformationMatrix,
) -> NDArray[Shape["H, W, ..."], Float32]:
    xyz = camera_vectors * depth

    projected_points = project_points(
        points=xyz.reshape(-1, 3),
        lens_model=lens_model,
        transformation_matrix=transformation_matrix.inverse(),
    ).reshape(*camera_vectors.shape[:2], 2)

    return np.stack(
        [
            map_coordinates(
                input=_image,
                coordinates=[projected_points[..., 1], projected_points[..., 0]],
                order=1,
                mode="constant",
                cval=np.nan,
            )
            for _image in image.transpose(2, 0, 1)
        ],
        axis=-1,
        dtype=np.float32,
    )


def plane_sweeping(  # type: ignore
    image: NDArray[Shape["H, W, ..."], Float32],
    lens_model: LensModel,
    secondary_images: list[NDArray[Shape["H, W, ..."], Float32]],
    secondary_lens_models: list[LensModel],
    secondary_transformation_matrices: list[TransformationMatrix],
    depth_range: NDArray[Shape["2"], Float32],
    step_size: float,
    block_size: int,
    subpixel_fit: bool = True,
) -> NDArray[Shape["H, W, 3"], Float32]: ...
