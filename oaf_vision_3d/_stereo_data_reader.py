from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d.lens_model import CameraMatrix, LensModel
from oaf_vision_3d.transformation_matrix import TransformationMatrix


@dataclass
class StereoData:
    image_0: NDArray[Shape["H, W, 3"], Float32]
    image_1: NDArray[Shape["H, W, 3"], Float32]
    lens_model_0: LensModel
    lens_model_1: LensModel
    transformation_matrix: TransformationMatrix
    expected_disparity: NDArray[Shape["2"], Float32]
    width: int
    height: int
    ground_truth_disparity: Optional[NDArray[Shape["H, W"], Float32]] = None

    @staticmethod
    def from_path(data_dir: Path) -> StereoData:
        if (data_dir / "image_0.png").exists():
            return _parse_dataset_custom(data_dir)
        if (data_dir / "im0.png").exists():
            return _parse_dataset_mobile(data_dir)
        raise NotImplementedError("Unknown dataset format.")


def _parse_dataset_custom(data_dir: Path) -> StereoData:
    image_0 = plt.imread(data_dir / "image_0.png")[..., :3]
    image_1 = plt.imread(data_dir / "image_1.png")[..., :3]

    lens_model_0 = LensModel.read_from_json(data_dir / "lens_model_0.json")
    lens_model_1 = LensModel.read_from_json(data_dir / "lens_model_1.json")
    transformation_matrix = TransformationMatrix.read_from_json(
        data_dir / "transformation_matrix.json"
    )
    expected_disparity = np.load(data_dir / "expected_disparity.npy", allow_pickle=True)

    return StereoData(
        image_0=image_0,
        image_1=image_1,
        lens_model_0=lens_model_0,
        lens_model_1=lens_model_1,
        transformation_matrix=transformation_matrix,
        expected_disparity=expected_disparity,
        width=image_0.shape[1],
        height=image_0.shape[0],
    )


def _load_pfm(file_path: Path) -> NDArray[Shape["H, W"], Float32]:
    with open(file_path, "rb") as f:
        header = f.readline().decode().strip()
        if header not in {"PF", "Pf"}:
            raise ValueError("Not a valid PFM file.")

        dimensions = f.readline().decode().strip()
        width, height = map(int, dimensions.split())

        scale = float(f.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        data_type = endian + "f"

        disparity_data = np.fromfile(f, data_type).reshape((height, width))
        disparity_data = np.flipud(disparity_data)
        return disparity_data


def _parse_dataset_mobile(data_dir: Path) -> StereoData:
    image_0 = plt.imread(data_dir / "im0.png")[..., :3]
    image_1 = plt.imread(data_dir / "im1.png")[..., :3]
    ground_truth_disparity = _load_pfm(data_dir / "disp0.pfm")

    ground_truth_disparity[~np.isfinite(ground_truth_disparity)] = np.nan

    with open(data_dir / "calib.txt", encoding="utf-8") as f:
        calibration_data = f.readlines()
    calibration_data_splitted = [
        d.replace("\n", "").split("=") for d in calibration_data
    ]
    calibration_data_dict = {d[0]: d[1] for d in calibration_data_splitted}

    lens_model_0 = LensModel(
        camera_matrix=CameraMatrix.from_matrix(
            np.array(
                [
                    float(d)
                    for d in calibration_data_dict["cam0"]
                    .replace("[", "")
                    .replace("]", "")
                    .replace(";", "")
                    .split(" ")
                ],
                dtype=np.float32,
            ).reshape(3, 3)
        )
    )
    lens_model_1 = LensModel(
        camera_matrix=CameraMatrix.from_matrix(
            np.array(
                [
                    float(d)
                    for d in calibration_data_dict["cam1"]
                    .replace("[", "")
                    .replace("]", "")
                    .replace(";", "")
                    .split(" ")
                ],
                dtype=np.float32,
            ).reshape(3, 3)
        )
    )
    transformation_matrix = TransformationMatrix(
        translation=np.array(
            [float(calibration_data_dict["baseline"]), 0.0, 0.0], dtype=np.float32
        )
    )
    width = int(calibration_data_dict["width"])
    height = int(calibration_data_dict["height"])
    vmin = int(calibration_data_dict["vmin"])
    vmax = int(calibration_data_dict["vmax"])

    return StereoData(
        image_0=image_0,
        image_1=image_1,
        lens_model_0=lens_model_0,
        lens_model_1=lens_model_1,
        transformation_matrix=transformation_matrix,
        expected_disparity=np.sort(np.array([vmin, vmax], dtype=np.float32)),
        width=width,
        height=height,
        ground_truth_disparity=ground_truth_disparity,
    )
