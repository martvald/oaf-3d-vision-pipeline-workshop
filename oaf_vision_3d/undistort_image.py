# %% [markdown]
# # Undistort Image
#
# This function undistorts an image using a new camera matrix as done in the
# [workshop](../workshops/03_image_distortion_and_undistortion.ipynb).

# %%
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import CameraMatrix, LensModel


def undistort_image_with_new_camera_matrix(  # type: ignore
    image: NDArray[Shape["H, W, 3"], Float32],
    lens_model: LensModel,
    new_camera_matrix: CameraMatrix,
) -> NDArray[Shape["H, W, 3"], Float32]: ...
