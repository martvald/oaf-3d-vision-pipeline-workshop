# %% [markdown]
# # Lens Model
#
# This module implement the [OpenCV lens model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
# for camera calibration. The exploation of this was explored in
# [2: Understanding Camera Models](../workshops/02_understanding_camera_models.ipynb).
#
# ## Imports

# %%
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from nptyping import Float32, NDArray, Shape

# %% [markdown]
# ## Camera Matrix


# %%
@dataclass
class CameraMatrix:
    fx: float
    fy: float
    cx: float
    cy: float

    def focal_length(self) -> NDArray[Shape["2"], Float32]:
        return np.array([self.fx, self.fy], dtype=np.float32)

    def principal_point(self) -> NDArray[Shape["2"], Float32]:
        return np.array([self.cx, self.cy], dtype=np.float32)

    def as_matrix(self) -> NDArray[Shape["3, 3"], Float32]:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_matrix(matrix: NDArray[Shape["3, 3"], Float32]) -> CameraMatrix:
        return CameraMatrix(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    @staticmethod
    def from_dict(data: dict) -> CameraMatrix:
        return CameraMatrix(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )


def normalize_pixels(
    pixels: NDArray[Shape["H, W, 2"], Float32],
    camera_matrix: CameraMatrix,
) -> NDArray[Shape["H, W, 2"], Float32]:
    return (
        pixels - camera_matrix.principal_point()[None, None, :]
    ) / camera_matrix.focal_length()[None, None, :]


def _denormalize_pixels(
    pixels: NDArray[Shape["H, W, 2"], Float32],
    camera_matrix: CameraMatrix,
) -> NDArray[Shape["H, W, 2"], Float32]:
    return (
        pixels * camera_matrix.focal_length()[None, None, :]
    ) + camera_matrix.principal_point()[None, None, :]


# %% [markdown]
# ## Distortion Coefficients


# %%
@dataclass
class DistortionCoefficients:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    tau_x: float = 0.0
    tau_y: float = 0.0

    def as_opencv_vector(self) -> NDArray[Shape["14"], Float32]:
        return np.array(
            [
                self.k1,
                self.k2,
                self.p1,
                self.p2,
                self.k3,
                self.k4,
                self.k5,
                self.k6,
                self.s1,
                self.s2,
                self.s3,
                self.s4,
                self.tau_x,
                self.tau_y,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "k4": self.k4,
            "k5": self.k5,
            "k6": self.k6,
            "p1": self.p1,
            "p2": self.p2,
            "s1": self.s1,
            "s2": self.s2,
            "s3": self.s3,
            "s4": self.s4,
            "tau_x": self.tau_x,
            "tau_y": self.tau_y,
        }

    @staticmethod
    def from_dict(data: dict) -> DistortionCoefficients:
        return DistortionCoefficients(
            k1=float(data["k1"]),
            k2=float(data["k2"]),
            k3=float(data["k3"]),
            k4=float(data["k4"]),
            k5=float(data["k5"]),
            k6=float(data["k6"]),
            p1=float(data["p1"]),
            p2=float(data["p2"]),
            s1=float(data["s1"]),
            s2=float(data["s2"]),
            s3=float(data["s3"]),
            s4=float(data["s4"]),
            tau_x=float(data["tau_x"]),
            tau_y=float(data["tau_y"]),
        )


def _distort_pixels(
    normalized_pixels: NDArray[Shape["H, W, 2"], Float32],
    distortion_coefficients: DistortionCoefficients,
) -> NDArray[Shape["H, W, 2"], Float32]:
    return normalized_pixels


def _undistort_pixels(
    normalized_pixels: NDArray[Shape["H, W, 2"], Float32],
    distortion_coefficients: DistortionCoefficients,
    number_of_iterations: int = 10,
) -> NDArray[Shape["H, W, 2"], Float32]:
    undistorted_normalized_pixels = normalized_pixels.copy()
    for _ in range(number_of_iterations):
        undistorted_normalized_pixels += normalized_pixels - _distort_pixels(
            undistorted_normalized_pixels,
            distortion_coefficients=distortion_coefficients,
        )
    return undistorted_normalized_pixels


# %% [markdown]
# ## Lens Model


# %%
@dataclass
class LensModel:
    camera_matrix: CameraMatrix
    distortion_coefficients: DistortionCoefficients = field(
        default_factory=DistortionCoefficients
    )

    def normalize_pixels(
        self, pixels: NDArray[Shape["H, W, 2"], Float32]
    ) -> NDArray[Shape["H, W, 2"], Float32]:
        return normalize_pixels(pixels, self.camera_matrix)

    def denormalize_pixels(
        self, pixels: NDArray[Shape["H, W, 2"], Float32]
    ) -> NDArray[Shape["H, W, 2"], Float32]:
        return _denormalize_pixels(pixels, self.camera_matrix)

    def distort_pixels(
        self, normalized_pixels: NDArray[Shape["H, W, 2"], Float32]
    ) -> NDArray[Shape["H, W, 2"], Float32]:
        return _distort_pixels(normalized_pixels, self.distortion_coefficients)

    def undistort_pixels(
        self, normalized_pixels: NDArray[Shape["H, W, 2"], Float32]
    ) -> NDArray[Shape["H, W, 2"], Float32]:
        return _undistort_pixels(normalized_pixels, self.distortion_coefficients)

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.to_dict(),
            "distortion_coefficients": self.distortion_coefficients.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> LensModel:
        return LensModel(
            camera_matrix=CameraMatrix.from_dict(data["camera_matrix"]),
            distortion_coefficients=DistortionCoefficients.from_dict(
                data["distortion_coefficients"]
            ),
        )

    def write_to_json(self, file_path: Path) -> None:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    @staticmethod
    def read_from_json(file_path: Path) -> LensModel:
        with file_path.open("r", encoding="utf-8") as file:
            return LensModel.from_dict(json.load(file))
