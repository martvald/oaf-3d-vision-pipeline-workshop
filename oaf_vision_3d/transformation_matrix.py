# %% [markdown]
# # Transformation Matrix
#
# This class holds a 4x4 transformation matrix, which is a combination of a 3x3
# rotation matrix and a 3x1 translation vector. It uses the
# `scipy.spatial.transform.Rotation` class to represent the rotation, as this class
# provides a convenient way work with different rotation representations (e.g. rotation
# matrix, quaternion, Euler angles, etc.), and numpy for the translation vector.
#
# The class provides methods to:
# - Convert to and from a 4x4 transformation matrix
# - Create a transformation from a rotation vector and translation vector
# - Invert the transformation
# - Rotate, translate, and transform points
# - Use the `@` operator to apply the transformation to points or combine two
#   transformations
# - Convert to and from a dictionary
# - Write to and read from a JSON file

# %%
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

import numpy as np
from nptyping import Float32, NDArray, Shape
from scipy.spatial.transform import Rotation


@dataclass
class TransformationMatrix:
    rotation: Rotation = Rotation.identity()
    translation: NDArray[Shape["3"], Float32] = field(
        default_factory=lambda: np.array([0, 0, 0], np.float32)
    )

    def as_matrix(self) -> NDArray[Shape["4, 4"], Float32]:
        matrix = np.identity(4, np.float32)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix

    @staticmethod
    def from_matrix(matrix: NDArray[Shape["4, 4"], Float32]) -> TransformationMatrix:
        return TransformationMatrix(
            rotation=Rotation.from_matrix(matrix[:3, :3]), translation=matrix[:3, 3]
        )

    @staticmethod
    def from_rvec_and_tvec(
        rvec: NDArray[Shape["3"], Float32], tvec: NDArray[Shape["3"], Float32]
    ) -> TransformationMatrix:
        return TransformationMatrix(
            rotation=Rotation.from_rotvec(rvec), translation=tvec
        )

    def inverse(self) -> TransformationMatrix:
        return TransformationMatrix.from_matrix(np.linalg.inv(self.as_matrix()))

    def rotate(
        self, points: NDArray[Shape["H, W, 3"], Float32]
    ) -> NDArray[Shape["H, W, 3"], Float32]:
        return np.einsum("ij,...j->...i", self.rotation.as_matrix(), points)

    def translate(
        self, points: NDArray[Shape["H, W, 3"], Float32]
    ) -> NDArray[Shape["H, W, 3"], Float32]:
        return points + self.translation[None, None, :]

    def transform(
        self, points: NDArray[Shape["H, W, 3"], Float32]
    ) -> NDArray[Shape["H, W, 3"], Float32]:
        return self.translate(points=self.rotate(points=points))

    @overload
    def __matmul__(
        self, other: NDArray[Shape["H, W, 3"], Float32]
    ) -> NDArray[Shape["H, W, 3"], Float32]: ...

    @overload
    def __matmul__(self, other: TransformationMatrix) -> TransformationMatrix: ...

    def __matmul__(
        self, other: NDArray[Shape["H, W, 3"], Float32] | TransformationMatrix
    ) -> NDArray[Shape["H, W, 3"], Float32] | TransformationMatrix:
        if isinstance(other, NDArray):
            return self.transform(points=other)
        if isinstance(other, TransformationMatrix):
            return TransformationMatrix.from_matrix(
                self.as_matrix() @ other.as_matrix()
            )
        raise NotImplementedError(other)

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> TransformationMatrix:
        return TransformationMatrix(
            rotation=Rotation.from_quat(np.array(data["rotation"], dtype=np.float32)),
            translation=np.array(data["translation"], dtype=np.float32),
        )

    def write_to_json(self, file_path: Path) -> None:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    @staticmethod
    def read_from_json(file_path: Path) -> TransformationMatrix:
        with file_path.open("r", encoding="utf-8") as file:
            return TransformationMatrix.from_dict(json.load(file))
