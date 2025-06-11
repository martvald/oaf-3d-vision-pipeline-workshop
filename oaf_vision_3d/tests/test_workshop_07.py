import numpy as np

from oaf_vision_3d._stereo_data_reader import StereoData
from oaf_vision_3d.plane_sweeping import plane_sweeping
from oaf_vision_3d.tests.status import Status
from test_data.data_paths import DataPaths


def workshop_07_results(overwrite: bool = False) -> Status:
    stereo_data_0 = StereoData.from_path(DataPaths.stereo_data_0_dir)
    stereo_data_1 = StereoData.from_path(DataPaths.stereo_data_1_dir)

    xyz = plane_sweeping(
        image=stereo_data_0.image_0[:500, :800],
        lens_model=stereo_data_0.lens_model_0,
        secondary_images=[
            stereo_data_0.image_1[:500, :800],
            stereo_data_1.image_1[:500, :800],
        ],
        secondary_lens_models=[stereo_data_0.lens_model_1, stereo_data_1.lens_model_1],
        secondary_transformation_matrices=[
            stereo_data_0.transformation_matrix,
            stereo_data_1.transformation_matrix,
        ],
        depth_range=np.array([135.0, 142.0], dtype=np.float32),
        step_size=0.5,
        block_size=5,
        subpixel_fit=True,
    )

    if xyz is None:
        return Status.NOT_STARTED

    if overwrite:
        np.savez(DataPaths.ws_07_reference_data, z=xyz[:, :, 2])

    reference_data = np.load(DataPaths.ws_07_reference_data)

    test_result = np.allclose(
        xyz[:, :, 2], reference_data["z"], atol=1e-3, equal_nan=True
    )

    return Status.from_bool(test_result)


def test_workshop_07(overwrite: bool = False) -> None:
    assert workshop_07_results(overwrite=overwrite)


if __name__ == "__main__":
    test_workshop_07(overwrite=False)
