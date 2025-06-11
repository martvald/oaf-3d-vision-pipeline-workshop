# %% [markdown]
# # Solve PnP
#
# This function solves the PnP problem as done in the
# [workshop](../workshops/04_3d_2d_projections_and_pnp.ipynb).

# %%
import cv2
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import LensModel


def project_points_with_jacobian(
    points: NDArray[Shape["*, 3"], Float32],
    rvec: NDArray[Shape["3"], Float32],
    tvec: NDArray[Shape["3"], Float32],
    lens_model: LensModel,
) -> tuple[NDArray[Shape["*, 2"], Float32], NDArray[Shape["*, 2, 6"], Float32]]:
    projected_points, jacobian = cv2.projectPoints(
        objectPoints=points,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=lens_model.camera_matrix.as_matrix(),
        distCoeffs=lens_model.distortion_coefficients.as_opencv_vector(),
    )
    return (
        projected_points.astype(np.float32)[:, 0, :],
        jacobian.astype(np.float32)[:, :6].reshape(-1, 2, 6),
    )


def solve_pnp(  # type: ignore
    points: NDArray[Shape["*, 3"], Float32],
    pixels: NDArray[Shape["*, 2"], Float32],
    lens_model: LensModel,
    rvec: NDArray[Shape["3"], Float32] = np.zeros(3, dtype=np.float32),
    tvec: NDArray[Shape["3"], Float32] = np.zeros(3, dtype=np.float32),
    epsilon: float = 1e-5,
    max_iterations: int = 100,
) -> tuple[NDArray[Shape["3"], Float32], NDArray[Shape["3"], Float32]]: ...
