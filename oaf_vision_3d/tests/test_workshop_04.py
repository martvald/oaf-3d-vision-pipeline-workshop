import numpy as np

from oaf_vision_3d.lens_model import CameraMatrix, DistortionCoefficients, LensModel
from oaf_vision_3d.project_points import project_points
from oaf_vision_3d.solve_pnp import solve_pnp
from oaf_vision_3d.tests.status import Status
from oaf_vision_3d.transformation_matrix import TransformationMatrix


def workshop_04_results() -> Status:
    lens_model = LensModel(
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

    points = np.stack(
        [
            *np.meshgrid(np.linspace(-20, 20, 10), np.linspace(-20, 20, 10)),
            np.zeros((10, 10)),
        ],
        axis=-1,
        dtype=np.float32,
    )

    rvec_to_find = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    tvec_to_find = np.array([5.6, -4.5, 98.7], dtype=np.float32)

    projected_pixels = project_points(
        points=points.reshape(-1, 3),
        lens_model=lens_model,
        transformation_matrix=TransformationMatrix.from_rvec_and_tvec(
            rvec=rvec_to_find, tvec=tvec_to_find
        ),
    ).reshape(10, 10, 2)

    rvec_tvec = solve_pnp(
        points=points.reshape(-1, 3),
        pixels=projected_pixels.reshape(-1, 2),
        lens_model=lens_model,
    )

    if rvec_tvec is None:
        return Status.NOT_STARTED

    test_result = np.allclose(rvec_tvec[0], rvec_to_find, atol=1e-3) and np.allclose(
        rvec_tvec[1], tvec_to_find, atol=1e-3
    )

    return Status.from_bool(test_result)


def test_workshop_04() -> None:
    assert workshop_04_results()


if __name__ == "__main__":
    test_workshop_04()
