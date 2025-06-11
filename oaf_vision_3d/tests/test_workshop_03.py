import numpy as np
from matplotlib import pyplot as plt

from oaf_vision_3d.lens_model import CameraMatrix, LensModel
from oaf_vision_3d.tests.status import Status
from oaf_vision_3d.undistort_image import undistort_image_with_new_camera_matrix
from test_data.data_paths import DataPaths


def workshop_03_results(overwrite: bool = False) -> Status:
    image_house = plt.imread(DataPaths.distorted_house)
    lens_model_house = LensModel.read_from_json(DataPaths.distorted_house_lens_model)

    scales = [0.75, 1.0, 1.25]
    undistorted_images = []
    for scale in scales:
        new_camera_matrix = CameraMatrix(
            fx=lens_model_house.camera_matrix.fx * scale,
            fy=lens_model_house.camera_matrix.fy * scale,
            cx=lens_model_house.camera_matrix.cx,
            cy=lens_model_house.camera_matrix.cy,
        )
        undistorted_images.append(
            undistort_image_with_new_camera_matrix(
                image=image_house,
                lens_model=lens_model_house,
                new_camera_matrix=new_camera_matrix,
            )
        )

    if any(undistorted_image is None for undistorted_image in undistorted_images):
        return Status.NOT_STARTED

    if overwrite:
        np.savez(DataPaths.ws_03_reference_data, undistorted_images=undistorted_images)

    reference_data = np.load(DataPaths.ws_03_reference_data)

    test_result = np.allclose(
        undistorted_images,
        reference_data["undistorted_images"],
        atol=1e-5,
    )

    return Status.from_bool(test_result)


def test_workshop_03(overwrite: bool = False) -> None:
    assert workshop_03_results(overwrite=overwrite)


if __name__ == "__main__":
    test_workshop_03(overwrite=False)
