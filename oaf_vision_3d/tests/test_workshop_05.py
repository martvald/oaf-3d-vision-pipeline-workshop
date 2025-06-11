from copy import deepcopy

import numpy as np

from oaf_vision_3d.lens_model import CameraMatrix, DistortionCoefficients, LensModel
from oaf_vision_3d.project_points import project_points
from oaf_vision_3d.tests.status import Status
from oaf_vision_3d.transformation_matrix import TransformationMatrix
from oaf_vision_3d.triangulation import triangulate_points


def workshop_05_results() -> Status:
    lens_model_0 = LensModel(
        camera_matrix=CameraMatrix(
            fx=2500.0,
            fy=2500.0,
            cx=1250.0,
            cy=1000.0,
        ),
        distortion_coefficients=DistortionCoefficients(
            k1=0.3,
            k2=-0.1,
            p1=-0.02,
        ),
    )
    lens_model_1 = deepcopy(lens_model_0)

    points = np.stack(
        [
            *np.meshgrid(np.linspace(-20, 20, 10), np.linspace(-20, 20, 10)),
            100 * np.ones((10, 10)),
        ],
        axis=-1,
        dtype=np.float32,
    )

    pixels_0 = project_points(
        points=points.reshape(-1, 3),
        lens_model=lens_model_0,
    ).reshape(10, 10, 2)

    rvec = np.array([-0.002, -0.15, 0.001], dtype=np.float32)
    tvec = np.array([15.0, 0.2, 2.1], dtype=np.float32)
    transformation_matrix = TransformationMatrix.from_rvec_and_tvec(
        rvec=rvec, tvec=tvec
    )

    pixels_1 = project_points(
        points=points.reshape(-1, 3),
        lens_model=lens_model_1,
        transformation_matrix=transformation_matrix.inverse(),
    ).reshape(10, 10, 2)

    undistorted_normalized_pixels_0 = lens_model_0.undistort_pixels(
        lens_model_0.normalize_pixels(pixels=pixels_0)
    )
    undistorted_normalized_pixels_1 = lens_model_1.undistort_pixels(
        lens_model_1.normalize_pixels(pixels=pixels_1)
    )

    trianguated_points = triangulate_points(
        undistorted_normalized_pixels_0=undistorted_normalized_pixels_0,
        undistorted_normalized_pixels_1=undistorted_normalized_pixels_1,
        transformation_matrix=transformation_matrix,
    )

    if trianguated_points is None:
        return Status.NOT_STARTED

    test_result = np.allclose(points, trianguated_points, atol=1e-5)

    return Status.from_bool(test_result)


def test_workshop_05() -> None:
    assert workshop_05_results()


if __name__ == "__main__":
    test_workshop_05()
