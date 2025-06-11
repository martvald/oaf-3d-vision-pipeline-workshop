import numpy as np

from oaf_vision_3d.lens_model import CameraMatrix, DistortionCoefficients, LensModel
from oaf_vision_3d.tests.status import Status
from test_data.data_paths import DataPaths


def workshop_02_results(overwrite: bool = False) -> Status:
    pixels = np.indices((201, 251), dtype=np.float32).transpose(1, 2, 0) * 10
    lens_models = [
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(k1=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(k2=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(k3=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(k4=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(k5=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(k6=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(p1=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(p2=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(s1=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(s2=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(s3=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(s4=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(tau_x=0.1),
        ),
        LensModel(
            camera_matrix=CameraMatrix(fx=2500.0, fy=2500.0, cx=1250, cy=1000),
            distortion_coefficients=DistortionCoefficients(tau_y=0.1),
        ),
    ]

    normalized_pixels = [
        lens_model.normalize_pixels(pixels=pixels) for lens_model in lens_models
    ]
    distorted_pixels = [
        lens_model.distort_pixels(normalized_pixels=normalized_pixel)
        for lens_model, normalized_pixel in zip(lens_models, normalized_pixels)
    ]

    if np.all(
        [
            np.all(distorted_pixel == normalized_pixel)
            for distorted_pixel, normalized_pixel in zip(
                distorted_pixels, normalized_pixels
            )
        ]
    ):
        return Status.NOT_STARTED

    if overwrite:
        reference_data = np.array(distorted_pixels)
        np.savez(DataPaths.ws_02_reference_data, distorted_pixels=reference_data)

    reference_data = np.load(DataPaths.ws_02_reference_data)

    test_result = np.allclose(
        distorted_pixels,
        reference_data["distorted_pixels"],
        atol=1e-5,
    )

    return Status.from_bool(test_result)


def test_workshop_02(overwrite: bool = False) -> None:
    assert workshop_02_results(overwrite=overwrite)


if __name__ == "__main__":
    test_workshop_02(overwrite=False)
